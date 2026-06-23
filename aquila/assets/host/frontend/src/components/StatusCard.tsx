import { Box, Typography, useTheme } from "@mui/material";

type StatusCardProps = {
  label: string;
  value: string | number;
  // Semantic accent resolved from the theme palette — no hex literals.
  accent?: "success" | "warning";
};

export function StatusCard({ label, value, accent }: StatusCardProps) {
  const theme = useTheme();
  // Color only where it means something; neutral cards get a quiet dot.
  const dotColor = accent
    ? theme.palette[accent].main
    : theme.palette.divider;
  return (
    <Box className="status-card">
      <Box className="status-label-row">
        <Box className="status-dot" sx={{ bgcolor: dotColor }} />
        <Typography className="status-label">{label}</Typography>
      </Box>
      <Typography className="status-value">{value}</Typography>
    </Box>
  );
}
