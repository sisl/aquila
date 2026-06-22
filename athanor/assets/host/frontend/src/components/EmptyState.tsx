import { ReactNode } from "react";
import { Box, SvgIconProps, Typography } from "@mui/material";

type EmptyStateProps = {
  // An @mui/icons-material component, rendered small and faint.
  icon?: React.ComponentType<SvgIconProps>;
  primary: ReactNode;
  hint?: ReactNode;
};

// The one empty-state treatment: quiet, centered, with an optional hint
// telling the user where the data will come from.
export function EmptyState({ icon: Icon, primary, hint }: EmptyStateProps) {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 0.75,
        py: 5,
        textAlign: "center"
      }}
    >
      {Icon && <Icon sx={{ fontSize: 22, color: "text.secondary", opacity: 0.5 }} />}
      <Typography variant="body2" className="muted">
        {primary}
      </Typography>
      {hint && (
        <Typography variant="caption" className="muted" sx={{ opacity: 0.8 }}>
          {hint}
        </Typography>
      )}
    </Box>
  );
}
