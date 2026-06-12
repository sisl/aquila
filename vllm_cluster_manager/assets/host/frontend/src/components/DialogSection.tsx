import type { ReactNode } from "react";
import { Box, Divider, Typography } from "@mui/material";

type DialogSectionProps = {
  title: string;
  // Muted one-line explainer under the title.
  hint?: ReactNode;
  // Right-aligned control (button cluster, select) on the title row.
  action?: ReactNode;
  // Suppress the leading divider (for the first section in a dialog).
  first?: boolean;
  children?: ReactNode;
};

// One section grammar for multi-section dialogs (Manage, Settings): hairline
// divider between sections, confident title with optional hint, optional
// right-aligned action — mirroring the panel-header language of the app.
export function DialogSection({
  title,
  hint,
  action,
  first = false,
  children
}: DialogSectionProps) {
  return (
    <Box>
      {!first && <Divider sx={{ my: 2.5 }} />}
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: 1.5
        }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h6">{title}</Typography>
          {hint && (
            <Typography variant="body2" className="muted">
              {hint}
            </Typography>
          )}
        </Box>
        {action && <Box sx={{ flexShrink: 0 }}>{action}</Box>}
      </Box>
      {children && <Box sx={{ mt: 1.5 }}>{children}</Box>}
    </Box>
  );
}
