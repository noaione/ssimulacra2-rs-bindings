fn main() {
    let out_dir = build_fssimu2();

    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // Force static linking
    println!("cargo:rustc-link-lib=static=ssimu2");

    // Required system libs (platform-dependent)
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=ucrt");
    }
}

fn build_fssimu2() -> std::path::PathBuf {
    use std::path::Path;
    use std::process::Command;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let optimize = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "ReleaseFast"
    };

    // fssimu2 is a zig project, so we can use zig build to compile it
    let status = Command::new("zig")
        .arg("build")
        .arg(format!("-Doptimize={optimize}"))
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to build fssimu2 with zig");

    if !status.success() {
        panic!("Failed to build fssimu2");
    }

    let out_dir = Path::new(&manifest_dir).join("zig-out").join("lib");
    out_dir
}
