{
  description = "Scrumpler - Audio sample processor for experimental music production";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python = pkgs.python3;
        pythonPackages = python.pkgs;

        # Python dependencies
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          scipy
          soundfile
        ]);

      in {
        # Development shell - use with `nix develop`
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.libsndfile  # Required by soundfile
          ];

          shellHook = ''
            echo "Scrumpler development environment"
            echo "Run: python sample_processor.py --help"
            echo "     python batch_processor.py help"
          '';
        };

        # Package - use with `nix build`
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "scrumpler";
          version = "0.1.0";
          src = ./.;

          buildInputs = [ pythonEnv ];
          nativeBuildInputs = [ pkgs.makeWrapper ];

          installPhase = ''
            mkdir -p $out/bin $out/lib/scrumpler
            cp sample_processor.py batch_processor.py $out/lib/scrumpler/

            makeWrapper ${pythonEnv}/bin/python $out/bin/scrumpler \
              --add-flags "$out/lib/scrumpler/sample_processor.py" \
              --prefix LD_LIBRARY_PATH : ${pkgs.libsndfile}/lib

            makeWrapper ${pythonEnv}/bin/python $out/bin/scrumpler-batch \
              --add-flags "$out/lib/scrumpler/batch_processor.py" \
              --prefix LD_LIBRARY_PATH : ${pkgs.libsndfile}/lib
          '';
        };

        # App - use with `nix run`
        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/scrumpler";
        };

        apps.batch = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/scrumpler-batch";
        };
      }
    );
}
