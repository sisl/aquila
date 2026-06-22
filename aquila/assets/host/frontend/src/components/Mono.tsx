import { ReactNode } from "react";
import { Box, SxProps, Theme } from "@mui/material";

import { tokens } from "../styles/theme";

type MonoProps = {
  block?: boolean;
  children: ReactNode;
  sx?: SxProps<Theme>;
};

// The one monospace treatment: ids, image tags, args, code. Inline by
// default; `block` for standalone lines/areas.
export function Mono({ block, children, sx }: MonoProps) {
  return (
    <Box
      component={block ? "div" : "span"}
      sx={{
        fontFamily: tokens.fontMono,
        fontSize: "0.75rem",
        lineHeight: 1.6,
        fontVariantLigatures: "none",
        ...sx
      }}
    >
      {children}
    </Box>
  );
}
