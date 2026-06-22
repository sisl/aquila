import { Typography } from "@mui/material";
import type { SxProps, Theme } from "@mui/material/styles";
import type { ReactNode } from "react";

type SectionLabelProps = {
  children: ReactNode;
  sx?: SxProps<Theme>;
};

// Uppercase section label used across forms and panels; the actual style
// lives in the theme's `overline` typography variant.
export function SectionLabel({ children, sx }: SectionLabelProps) {
  return (
    <Typography variant="overline" sx={{ display: "block", ...sx }}>
      {children}
    </Typography>
  );
}
