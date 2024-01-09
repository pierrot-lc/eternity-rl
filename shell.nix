{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      CudaSupport = true;
    };
  }
}:
let
  python-packages = ps: with ps; [
    pip
    setuptools
    virtualenv
  ];
in
  (pkgs.buildFHSUserEnv {
    name = "pytorch";
    targetPkgs = pkgs: (with pkgs; [
      (python311.withPackages python-packages)
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      just
      pdm
    ]);
    runScript = "bash";
  }).env
