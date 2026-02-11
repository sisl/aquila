import { Box, Typography } from "@mui/material";

type StatusCardProps = {
  label: string;
  value: string | number;
  accent?: string;
};

export function StatusCard({ label, value, accent }: StatusCardProps) {
  return (
    <Box
      className="status-card status-accent"
      sx={{ borderLeftColor: accent ?? "#485b6e" }}
    >
      <Typography className="status-label">{label}</Typography>
      <Typography className="status-value">{value}</Typography>
    </Box>
  );
}
