use std::fs;
use std::path::{Path, PathBuf};

fn strip_val(s: &str) -> String {
    let t = s.split('#').next().unwrap_or(s).trim();
    t.trim_matches('"').trim_matches('\'').trim().to_string()
}

fn normalize_slug(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn find_ico_in_dir(dir: &Path, target_norm_stem: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut exact: Option<PathBuf> = None;
    let mut others: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file()
            && p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("ico"))
        {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if normalize_slug(stem) == target_norm_stem {
                exact = Some(p);
                break;
            }
            others.push(p);
        }
    }
    if exact.is_some() {
        return exact;
    }
    others.sort();
    others.into_iter().next()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));
    let id_file = manifest_dir.join("PROJECT_IDENTITY.txt");
    if id_file.is_file() {
        println!("cargo:rerun-if-changed=PROJECT_IDENTITY.txt");
    }
    let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "continuum".into());
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "1.0.0".into());

    let mut bin_name = String::from("Continuum");
    let mut app_name = String::from("Continuum");
    let mut author = String::from("gtaking");
    let mut company = String::from("ahkscriptlerim");
    let mut description =
        String::from("Continuum — Exact Arbitrary-Precision, Linear Algebra & Symbolic Engine");

    // 1. Parse `[package]`, `[[bin]]`, and `[package.metadata.commercial]` in Cargo.toml (error-proof fallback when PROJECT_IDENTITY.txt is absent)
    if let Ok(toml_txt) = fs::read_to_string(manifest_dir.join("Cargo.toml")) {
        let mut sec = String::new();
        let mut first_bin = String::new();
        for raw in toml_txt.lines() {
            let line = raw.trim();
            if line.starts_with('[') && line.ends_with(']') {
                sec = line.to_string();
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let val = strip_val(v);
                if val.is_empty() {
                    continue;
                }
                match sec.as_str() {
                    "[package]" => match k.trim() {
                        "default-run" => bin_name = val,
                        "description" => description = val,
                        _ => {}
                    },
                    "[[bin]]" => {
                        if k.trim() == "name" && first_bin.is_empty() {
                            first_bin = val;
                        }
                    }
                    "[package.metadata.commercial]" => match k.trim() {
                        "display_name" | "app_name" => app_name = val,
                        "author" | "author_name" => author = val,
                        "company_name" | "company" => company = val,
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        if bin_name.is_empty() && !first_bin.is_empty() {
            bin_name = first_bin;
        }
    }

    // 2. Optional `PROJECT_IDENTITY.txt` single-source override (100% optional: if absent, step 1 Cargo.toml values are used directly)
    if id_file.is_file()
        && let Ok(id_txt) = fs::read_to_string(&id_file)
    {
        for raw in id_txt.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let val = strip_val(v);
                if val.is_empty() {
                    continue;
                }
                match k.trim().to_ascii_lowercase().as_str() {
                    "project_name" | "app_name" | "display_name" | "name" => app_name = val,
                    "author" | "author_name" => author = val,
                    "company" | "company_name" | "institution" => company = val,
                    "description" => description = val,
                    _ => {}
                }
            }
        }
    }

    let publisher = format!("{author} <{company}>");
    let copyright = format!("Copyright © 2026 {author} <{company}>. All rights reserved.");
    let slug_clean: String = pkg_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let target_ico_norm = format!("{}icon", normalize_slug(&pkg_name));

    println!("cargo:rustc-env=APP_DISPLAY_NAME={app_name}");
    println!("cargo:rustc-env=APP_AUTHOR={author}");
    println!("cargo:rustc-env=APP_COMPANY={company}");
    println!("cargo:rustc-env=APP_PUBLISHER={publisher}");
    println!("cargo:rustc-env=APP_DESCRIPTION={description}");
    println!("cargo:rustc-env=APP_SLUG={slug_clean}");

    // 3. Dynamic `.ico` resolution:
    //    - If workspace root already has `<projectname>_icon.ico` or any `*.ico`, use it directly (`0` lookup into `manual release`).
    //    - Otherwise check `manual release\` for `*.ico` and copy it into workspace root as `<projectname>_icon.ico`.
    let resolved_ico = if let Some(ws_ico) = find_ico_in_dir(&manifest_dir, &target_ico_norm) {
        ws_ico
    } else {
        let mr_dir = manifest_dir.join("manual release");
        if let Some(mr_ico) = find_ico_in_dir(&mr_dir, &target_ico_norm) {
            let dst = manifest_dir.join(format!("{slug_clean}_icon.ico"));
            let _ = fs::copy(&mr_ico, &dst);
            if dst.is_file() { dst } else { mr_ico }
        } else {
            manifest_dir.join(format!("{slug_clean}_icon.ico"))
        }
    };

    if resolved_ico.is_file() {
        println!("cargo:rerun-if-changed={}", resolved_ico.display());
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "target".into()));
    let embedded_ico = out_dir.join("embedded_app_icon.ico");
    if resolved_ico.is_file() {
        let _ = fs::copy(&resolved_ico, &embedded_ico);
    } else {
        let _ = fs::write(&embedded_ico, b"");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows" {
        println!("cargo:rustc-link-arg-bin={bin_name}=/EXPORT:NvOptimusEnablement");
        println!("cargo:rustc-link-arg-bin={bin_name}=/EXPORT:AmdPowerXpressRequestHighPerformance");
        let mut res = winres::WindowsResource::new();
        if resolved_ico.is_file()
            && let Some(ico_str) = resolved_ico.to_str()
        {
            res.set_icon(ico_str);
        }
        res.set("FileDescription", &description);
        res.set("ProductName", &app_name);
        res.set("OriginalFilename", &format!("{bin_name}.exe"));
        res.set("InternalName", &bin_name);
        res.set("CompanyName", &company);
        res.set("LegalCopyright", &copyright);
        res.set_version_info(winres::VersionInfo::PRODUCTVERSION, 0x0001_0000_0000_0000);
        res.set_version_info(winres::VersionInfo::FILEVERSION, 0x0001_0000_0000_0000);

        let xml_esc = |s: &str| -> String {
            s.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;")
        };
        let app_xml = xml_esc(&app_name);
        let comp_xml = xml_esc(&company);
        let desc_xml = xml_esc(&description);

        let manifest_xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0" xmlns:asmv3="urn:schemas-microsoft-com:asm.v3">
    <assemblyIdentity
        version="{pkg_version}.0"
        processorArchitecture="*"
        name="{comp_xml}.{app_xml}"
        type="win32"
    />
    <description>{desc_xml}</description>
    <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
        <security>
            <requestedPrivileges>
                <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
            </requestedPrivileges>
        </security>
    </trustInfo>
    <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
        <application>
            <supportedOS Id="{{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}}"/>
            <supportedOS Id="{{1f676c76-80e1-4239-95bb-83d0f6d0da78}}"/>
            <supportedOS Id="{{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}}"/>
            <supportedOS Id="{{35138b9a-5d96-4fbd-8e2d-a2440225f93a}}"/>
        </application>
    </compatibility>
    <asmv3:application>
        <asmv3:windowsSettings>
            <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
            <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, PerMonitor</dpiAwareness>
            <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
            <activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>
        </asmv3:windowsSettings>
    </asmv3:application>
    <dependency>
        <dependentAssembly>
            <assemblyIdentity
                type="win32"
                name="Microsoft.Windows.Common-Controls"
                version="6.0.0.0"
                processorArchitecture="*"
                publicKeyToken="6595b64144ccf1df"
                language="*"
            />
        </dependentAssembly>
    </dependency>
</assembly>
"#
        );
        res.set_manifest(&manifest_xml);
        if let Err(e) = res.compile() {
            eprintln!("Warning: Failed to compile Windows resource manifest: {e}");
        }
    }
}
