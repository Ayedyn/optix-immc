#!/bin/sh
gdb --args ../../src/bin/mmc --compute optix -f optix.json -b 1 -F bin $@
