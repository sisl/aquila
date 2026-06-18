import { type MouseEvent, useState } from "react";

import ViewColumnOutlined from "@mui/icons-material/ViewColumnOutlined";
import {
  Box,
  Button,
  Checkbox,
  Divider,
  FormControlLabel,
  IconButton,
  Popover,
  Tooltip,
  Typography
} from "@mui/material";

import type { ColumnDef } from "../hooks/useColumnVisibility";
import { SectionLabel } from "./SectionLabel";

type ColumnPickerProps = {
  columns: ColumnDef[];
  userHidden: Set<string>;
  onToggle: (key: string) => void;
  onReset: () => void;
  isCustomized: boolean;
};

export function ColumnPicker({
  columns,
  userHidden,
  onToggle,
  onReset,
  isCustomized
}: ColumnPickerProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

  const open = Boolean(anchorEl);

  const handleOpen = (event: MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const toggleable = columns.filter((c) => !c.alwaysVisible);

  return (
    <>
      <Tooltip title="Columns" enterDelay={500}>
        <IconButton
          size="small"
          aria-label="Column visibility"
          onClick={handleOpen}
        >
          <ViewColumnOutlined sx={{ fontSize: 18, color: "text.secondary" }} />
        </IconButton>
      </Tooltip>
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{ paper: { sx: { p: 2, minWidth: 200 } } }}
      >
        <SectionLabel sx={{ mb: 0.5 }}>Columns</SectionLabel>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {toggleable.map((col) => {
            return (
              <FormControlLabel
                key={col.key}
                control={
                  <Checkbox
                    size="small"
                    checked={!userHidden.has(col.key)}
                    onChange={() => onToggle(col.key)}
                  />
                }
                label={
                  <Typography variant="body2">{col.label}</Typography>
                }
                sx={{ mr: 0 }}
              />
            );
          })}
        </Box>
        <Divider sx={{ my: 1 }} />
        <Button
          size="small"
          disabled={!isCustomized}
          onClick={() => {
            onReset();
          }}
          sx={{ textTransform: "none", fontWeight: 400 }}
        >
          Reset to default
        </Button>
      </Popover>
    </>
  );
}
