{pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = true;
  };
} }:
(pkgs.buildFHSUserEnv {
  name = "pytorch";
  targetPkgs = pkgs: (with pkgs; [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    # cudaPackages.nccl
    just
    python311
    python311Packages.pip
    python311Packages.virtualenv
  ]);
  runScript = "bash";
}).env
