#!/bin/zsh

WORK_DIR=$(dirname "$0")

source "$WORK_DIR/colors.sh"

GENERAL_HELP="${GREEN}Usage:${NC} ./run [--help] [<path>] [-c | --clean-run <path>]"

if [ "$2" = "-a" ]
  then
    echo -e "$GENERAL_HELP\n\n\n${GREEN}Commands and Options${NC}\n\n${GREEN}--help${NC} - Displays General usage\n\n${GREEN}<path>${NC} - Checks if Virtual Environment is present, if not sets up one and runs the program, not cleaning up the Virtual Environment\n\n${GREEN}-c | --clean-run <path>${NC} - The same as '<path>' but cleans up the Virtual Environment\n"

  else
    echo -e "$GENERAL_HELP\n
        For more information on command usage run './run --help -a'"
fi

