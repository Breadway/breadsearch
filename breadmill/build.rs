use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

const ORT_VERSION: &str = "1.28.0";
const ORT_TARBALL: &str = "onnxruntime-linux-x64-1.28.0.tgz";
const ORT_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz";
const ORT_SHA256: &str = "a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407";
const RUNTIME_FILES: [&str; 4] = [
    "libonnxruntime.so.1.28.0",
    "libonnxruntime_providers_shared.so",
    "LICENSE",
    "ThirdPartyNotices.txt",
];

fn main() {
    println!("cargo:rerun-if-env-changed=BREADMILL_ORT_TARBALL");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64")
    {
        panic!("breadmill's bundled ONNX Runtime currently supports x86_64 Linux only");
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let lib_dir = out.join("ort-runtime");
    fs::create_dir_all(&lib_dir).expect("create ORT runtime directory");

    let tarball = match env::var_os("BREADMILL_ORT_TARBALL") {
        Some(path) => {
            let path = PathBuf::from(path);
            verify_sha256(&fs::read(&path).expect("read BREADMILL_ORT_TARBALL"));
            path
        }
        None => {
            let path = out.join(ORT_TARBALL);
            if !path.exists() {
                let bytes = download(ORT_URL);
                verify_sha256(&bytes);
                fs::write(&path, &bytes).expect("write ORT tarball");
            } else {
                verify_sha256(&fs::read(&path).expect("read cached ORT tarball"));
            }
            path
        }
    };

    extract(&tarball, &lib_dir);
    let runtime_file = fs::read(lib_dir.join("libonnxruntime.so.1.28.0")).expect("read extracted ONNX Runtime library");
    let providers_file = fs::read(lib_dir.join("libonnxruntime_providers_shared.so"))
        .expect("read extracted ONNX Runtime providers library");
    for notice in ["LICENSE", "ThirdPartyNotices.txt"] {
        assert!(lib_dir.join(notice).is_file(), "ORT archive is missing {notice}");
    }
    fs::write(out.join("ort-runtime.bin"), &runtime_file).expect("write embedded ONNX Runtime library");
    fs::write(out.join("ort-runtime-providers.bin"), &providers_file)
        .expect("write embedded ONNX Runtime providers library");
    for notice in ["LICENSE", "ThirdPartyNotices.txt"] {
        fs::copy(lib_dir.join(notice), out.join(notice)).expect("copy ONNX Runtime notice");
    }
    for file in RUNTIME_FILES {
        assert!(lib_dir.join(file).is_file(), "ORT archive is missing {file}");
    }
    println!("cargo:rustc-env=BREADMILL_ORT_RUNTIME_LIB={}", out.join("ort-runtime.bin").display());
    println!("cargo:rustc-env=BREADMILL_ORT_RUNTIME_PROVIDERS={}", out.join("ort-runtime-providers.bin").display());
    println!("cargo:rustc-env=BREADMILL_ORT_LICENSE={}", out.join("LICENSE").display());
    println!("cargo:rustc-env=BREADMILL_ORT_NOTICES={}", out.join("ThirdPartyNotices.txt").display());
}

fn download(url: &str) -> Vec<u8> {
    let response = ureq::get(url).call().unwrap_or_else(|e| panic!("download {url}: {e}"));
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("read {url}: {e}"));
    bytes
}

fn verify_sha256(bytes: &[u8]) {
    let digest = Sha256::digest(bytes);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, ORT_SHA256, "ONNX Runtime {ORT_VERSION} archive checksum mismatch");
}

fn extract(tarball: &Path, dest: &Path) {
    let file = fs::File::open(tarball).expect("open ORT tarball");
    let mut archive = tar::Archive::new(GzDecoder::new(file));
    for entry in archive.entries().expect("read ORT tarball") {
        let mut entry = entry.expect("ORT tar entry");
        let path = entry.path().expect("ORT tar path").into_owned();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let is_library = RUNTIME_FILES[..2].contains(&name);
        let is_notice = RUNTIME_FILES[2..].contains(&name);
        if (is_library && path.parent().is_some_and(|p| p.ends_with("lib")))
            || (is_notice && path.parent().is_some_and(|p| p == Path::new("onnxruntime-linux-x64-1.28.0")))
        {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).expect("read ORT library");
            fs::write(dest.join(name), buf).expect("write ORT library");
        }
    }
    // ORT ships the main library as a versioned file. Expose the unversioned name
    // too, so dlopen of libonnxruntime.so and ORT_DYLIB_PATH both resolve.
    let so = dest.join("libonnxruntime.so.1.28.0");
    let link = dest.join("libonnxruntime.so");
    let _ = fs::remove_file(&link);
    std::os::unix::fs::symlink(&so, &link).expect("link libonnxruntime.so");
}
