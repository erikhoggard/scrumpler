{
  description = "Scrumpler - Audio sample processor for experimental music production";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # Pin to Python 3.11 for consistency with other projects
        python = pkgs.python311;

        # Python environment for development
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          scipy
          soundfile
          # Dev dependencies
          pytest
          black
          ruff
          hatchling
        ]);

        # The scrumpler package
        scrumpler = python.pkgs.buildPythonApplication {
          pname = "scrumpler";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            scipy
            soundfile
          ];

          # libsndfile is required at runtime
          makeWrapperArgs = [
            "--prefix" "LD_LIBRARY_PATH" ":" "${pkgs.libsndfile}/lib"
          ];

          meta = with pkgs.lib; {
            description = "Audio sample processor for experimental music production";
            license = licenses.mit;
            mainProgram = "scrumpler";
          };
        };

      in {
        packages = {
          default = scrumpler;
          inherit scrumpler;
        };

        # Development shell - use with `nix develop`
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.libsndfile
          ];

          shellHook = ''
            echo "Scrumpler development environment"
            echo ""
            echo "Commands:"
            echo "  scrumpler --help      # Main CLI"
            echo "  scrumpler-batch help  # Batch presets"
            echo ""
            echo "Or run directly:"
            echo "  python -m scrumpler.processor --help"
            echo "  python -m scrumpler.batch help"
          '';
        };

        # App - use with `nix run`
        apps.default = {
          type = "app";
          program = "${scrumpler}/bin/scrumpler";
        };

        apps.batch = {
          type = "app";
          program = "${scrumpler}/bin/scrumpler-batch";
        };
      }
    );
}
