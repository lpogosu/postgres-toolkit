"""Allow ``python -m pgtk`` as well as the installed ``pgtk`` script."""

from pgtk.cli import main

if __name__ == "__main__":
    main()
