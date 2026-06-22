import { alpha, createTheme } from "@mui/material/styles";

// Single source of truth for design tokens. global.css derives its CSS
// variables from these values; components must not hard-code hex colors —
// every derived shade below goes through alpha() on a token.
export const tokens = {
  ink: "#101214",
  // Secondary ink for outlined/ghost button labels.
  inkSoft: "#3f4753",
  muted: "#5d6570",
  accent: "#111827",
  accentHover: "#1f2937",
  // Two line weights are the whole border system: faint inside (rows,
  // dialog dividers), strong outside (panel borders, input outlines).
  line: "rgba(148, 163, 184, 0.45)",
  lineStrong: "rgba(148, 163, 184, 0.55)",
  background: "#f2f3f5",
  panel: "#ffffff",
  success: "#3d8b5e",
  warning: "#b8860b",
  error: "#c0392b",
  info: "#4a6fa5",
  // Darker shades for text on the soft tinted chip backgrounds.
  successText: "#2d6e4a",
  warningText: "#946a0c",
  infoText: "#3b5d8c",
  fontMono: "'SFMono-Regular', ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
  radius: 12,
  radiusLg: 16
};

const focusRing = {
  outline: `2px solid ${tokens.accent}`,
  outlineOffset: 2
};

const hoverTransition =
  "background-color 120ms ease, border-color 120ms ease, color 120ms ease";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: tokens.accent
    },
    secondary: {
      main: tokens.muted
    },
    success: {
      main: tokens.success
    },
    warning: {
      main: tokens.warning
    },
    error: {
      main: tokens.error
    },
    info: {
      main: tokens.info
    },
    divider: tokens.line,
    background: {
      default: tokens.background,
      paper: tokens.panel
    },
    text: {
      primary: tokens.ink,
      secondary: tokens.muted
    }
  },
  typography: {
    fontFamily: "'Source Sans 3', system-ui, sans-serif",
    h4: {
      fontWeight: 300,
      letterSpacing: "0.08em",
      textTransform: "uppercase"
    },
    h6: {
      fontSize: "1rem",
      fontWeight: 600,
      letterSpacing: "-0.01em"
    },
    button: {
      fontWeight: 400,
      textTransform: "none"
    },
    // Shared uppercase section-label style (see components/SectionLabel.tsx).
    overline: {
      fontSize: "0.7rem",
      fontWeight: 500,
      letterSpacing: "0.16em",
      textTransform: "uppercase",
      color: tokens.muted,
      lineHeight: 1.6
    }
  },
  shape: {
    borderRadius: tokens.radius
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          border: `1px solid ${tokens.lineStrong}`,
          boxShadow: "none"
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          border: `1px solid ${tokens.lineStrong}`,
          boxShadow: "none"
        }
      }
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: tokens.radiusLg
        }
      }
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: `1px solid ${tokens.line}`,
          padding: "10px 12px",
          fontSize: "0.8125rem"
        },
        head: {
          fontSize: "0.7rem",
          fontWeight: 500,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: tokens.muted,
          backgroundColor: tokens.panel,
          // Painted on the cell itself so the rule stays crisp while the
          // sticky header floats over scrolling rows (a plain border rides
          // along with the row below and visually detaches).
          borderBottom: "none",
          boxShadow: `inset 0 -1px 0 ${tokens.lineStrong}`,
          paddingTop: 6,
          paddingBottom: 8
        }
      }
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          transition: "background-color 120ms ease",
          "&.MuiTableRow-hover:hover": {
            backgroundColor: alpha(tokens.ink, 0.025)
          }
        }
      }
    },
    MuiChip: {
      defaultProps: {
        size: "small"
      },
      styleOverrides: {
        root: {
          fontWeight: 500,
          fontSize: "0.7rem",
          height: 20,
          borderRadius: 6,
          letterSpacing: "0.02em",
          lineHeight: 1,
          transition: hoverTransition,
          "&.Mui-focusVisible": focusRing,
          // colorDefault is not a typed override slot in MUI 5.
          "&.MuiChip-colorDefault": {
            backgroundColor: alpha(tokens.ink, 0.06),
            color: tokens.inkSoft
          },
          "& .MuiChip-label": {
            display: "flex",
            alignItems: "center"
          },
          "& .MuiChip-icon": {
            color: "inherit",
            marginLeft: 6
          }
        },
        // Soft tinted fills: status reads at a glance without competing
        // with buttons the way solid chips would.
        colorSuccess: {
          backgroundColor: alpha(tokens.success, 0.12),
          color: tokens.successText
        },
        colorWarning: {
          backgroundColor: alpha(tokens.warning, 0.14),
          color: tokens.warningText
        },
        colorError: {
          backgroundColor: alpha(tokens.error, 0.12),
          color: tokens.error
        },
        colorInfo: {
          backgroundColor: alpha(tokens.info, 0.12),
          color: tokens.infoText
        }
      }
    },
    MuiAlert: {
      defaultProps: {
        variant: "standard"
      },
      styleOverrides: {
        root: {
          border: "none",
          borderRadius: tokens.radius
        }
      }
    },
    MuiButton: {
      defaultProps: {
        disableElevation: true
      },
      styleOverrides: {
        root: {
          borderRadius: tokens.radius,
          transition: hoverTransition,
          "&.Mui-focusVisible": focusRing
        },
        // Ghost buttons (row-level actions): quiet muted text that
        // sharpens on hover; the data stays in the foreground.
        text: {
          color: tokens.muted,
          "&:hover": {
            color: tokens.ink,
            backgroundColor: alpha(tokens.ink, 0.04)
          },
          "&.MuiButton-colorError": {
            color: tokens.error,
            "&:hover": {
              color: tokens.error,
              backgroundColor: alpha(tokens.error, 0.06)
            }
          }
        },
        outlined: {
          borderColor: tokens.lineStrong,
          color: tokens.inkSoft,
          "&:hover": {
            borderColor: tokens.inkSoft,
            color: tokens.ink,
            backgroundColor: alpha(tokens.ink, 0.04)
          },
          "&.MuiButton-colorError": {
            borderColor: alpha(tokens.error, 0.45),
            color: tokens.error,
            "&:hover": {
              borderColor: tokens.error,
              color: tokens.error,
              backgroundColor: alpha(tokens.error, 0.06)
            }
          }
        },
        contained: {
          "&:hover": {
            backgroundColor: tokens.accentHover
          }
        },
        sizeSmall: {
          fontSize: "0.75rem",
          padding: "4px 10px"
        }
      }
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          transition: hoverTransition,
          "&.Mui-focusVisible": focusRing
        }
      }
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          transition: hoverTransition,
          "&.Mui-focusVisible": focusRing
        }
      }
    },
    MuiSkeleton: {
      defaultProps: {
        animation: "wave"
      },
      styleOverrides: {
        root: {
          backgroundColor: alpha(tokens.ink, 0.05)
        }
      }
    },
    // Slim form accordions (launch form optional sections).
    MuiAccordion: {
      defaultProps: {
        disableGutters: true,
        elevation: 0
      },
      styleOverrides: {
        root: {
          border: "none",
          backgroundColor: "transparent",
          "&:before": {
            display: "none"
          }
        }
      }
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: {
          padding: 0,
          minHeight: 40,
          "&.Mui-focusVisible": { backgroundColor: alpha(tokens.ink, 0.04) }
        },
        content: {
          margin: "8px 0",
          alignItems: "baseline",
          gap: 8
        }
      }
    },
    MuiAccordionDetails: {
      styleOverrides: {
        root: {
          padding: "0 0 16px"
        }
      }
    },
    MuiTabs: {
      styleOverrides: {
        root: { minHeight: 42 },
        indicator: { backgroundColor: tokens.accent, height: 2 }
      }
    },
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: 42,
          textTransform: "none",
          fontSize: "0.8125rem",
          fontWeight: 500,
          letterSpacing: "0.01em",
          color: tokens.muted,
          transition: hoverTransition,
          "&.Mui-selected": { color: tokens.ink },
          "&.Mui-focusVisible": focusRing
        }
      }
    },
    MuiTextField: {
      defaultProps: {
        size: "small"
      }
    },
    MuiSelect: {
      defaultProps: {
        size: "small"
      }
    },
    MuiInputBase: {
      styleOverrides: {
        root: {
          fontSize: "0.875rem"
        }
      }
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-notchedOutline": {
            borderColor: tokens.lineStrong
          },
          "&:hover .MuiOutlinedInput-notchedOutline": {
            borderColor: tokens.inkSoft
          },
          "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
            borderColor: tokens.accent,
            borderWidth: 1.5
          }
        },
        input: {
          "&.MuiInputBase-inputSizeSmall": {
            paddingTop: 8,
            paddingBottom: 8
          }
        }
      }
    },
    MuiFormHelperText: {
      styleOverrides: {
        root: {
          fontSize: "0.7rem",
          marginTop: 3,
          marginLeft: 2
        }
      }
    },
    MuiInputLabel: {
      styleOverrides: {
        root: {
          fontSize: "0.875rem"
        }
      }
    }
  }
});
