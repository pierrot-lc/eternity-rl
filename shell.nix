{pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = true;
  };
} }:
(pkgs.buildFHSUserEnv {
  name = "pytorch";
  targetPkgs = pkgs: (with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    # cudaPackages.nccl
  ]);
  runScript = "bash";
}).env
