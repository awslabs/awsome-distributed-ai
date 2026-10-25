# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Print usec/xfer from fi_pingpong client output. fi_pingpong prints a header row and then a data row, so locate the
# column by its name rather than by position.
BEGIN { c = -1 }
{
  for (i = 1; i <= NF; i++) if ($i == "usec/xfer") { c = i; next }
}
c > 0 && NF >= c { v = $c + 0; if (v > 0) { print v; exit } }
