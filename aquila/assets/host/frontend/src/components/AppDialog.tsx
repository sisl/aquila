import { ReactNode } from "react";
import {
  Breakpoint,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  SxProps,
  Theme,
  Typography,
  useMediaQuery,
  useTheme
} from "@mui/material";

type AppDialogProps = {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  // Muted caption rendered under the title (context like "Deployment #12").
  meta?: ReactNode;
  maxWidth?: Breakpoint;
  // Footer buttons, right-aligned. Convention: dismiss leftmost + outlined,
  // document actions between, the single primary action contained + last.
  actions?: ReactNode;
  paperClassName?: string;
  contentSx?: SxProps<Theme>;
  children: ReactNode;
};

// One dialog anatomy for the whole app: title (+ optional meta line),
// hairline-divided content, one right-aligned footer.
export function AppDialog({
  open,
  onClose,
  title,
  meta,
  maxWidth = "md",
  actions,
  paperClassName,
  contentSx,
  children
}: AppDialogProps) {
  const muiTheme = useTheme();
  const smallScreen = useMediaQuery(muiTheme.breakpoints.down("sm"));
  // Content-heavy dialogs take the whole phone screen; compact ones
  // (confirms, short forms) stay floating cards.
  const fullScreen = smallScreen && maxWidth !== "xs" && maxWidth !== "sm";

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      fullScreen={fullScreen}
      maxWidth={maxWidth}
      PaperProps={{ className: paperClassName ? `panel ${paperClassName}` : "panel" }}
    >
      <DialogTitle sx={{ p: { xs: "16px 14px 10px", sm: "20px 24px 12px" } }}>
        {title}
        {meta && (
          <Typography variant="caption" className="muted" sx={{ display: "block", mt: 0.25 }}>
            {meta}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent dividers sx={contentSx}>
        {children}
      </DialogContent>
      {actions && <DialogActions sx={{ p: { xs: "10px 14px 12px", sm: "12px 24px 16px" }, gap: 1 }}>{actions}</DialogActions>}
    </Dialog>
  );
}
