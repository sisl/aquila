import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#111827"
    },
    secondary: {
      main: "#6b7280"
    },
    background: {
      default: "#f5f6f7",
      paper: "#ffffff"
    },
    text: {
      primary: "#101214",
      secondary: "#6b7280"
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
      fontWeight: 500
    }
  },
  shape: {
    borderRadius: 16
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          border: "1px solid rgba(72, 91, 110, 0.18)",
          boxShadow: "none"
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          border: "1px solid rgba(72, 91, 110, 0.18)",
          boxShadow: "none"
        }
      }
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontSize: "0.7rem",
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "#5f6c7a"
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          borderRadius: 12,
          boxShadow: "none",
          fontWeight: 200,
          "&:hover": {
            backgroundColor: "transparent"
          },
          "&.MuiButton-outlinedInherit": {
            color: "#475569"
          },
          "&.MuiButton-outlinedInherit:hover": {
            color: "#334155 !important",
            borderColor: "#334155 !important",
            backgroundColor: "transparent !important"
          }
        },
        outlined: {
          borderColor: "rgba(148, 163, 184, 0.6)",
          color: "#475569",
          "&:hover": {
            backgroundColor: "transparent",
            borderColor: "#334155",
            color: "#334155 !important"
          }
        },
        outlinedInherit: {
          borderColor: "rgba(148, 163, 184, 0.6)",
          color: "#475569",
          "&:hover": {
            backgroundColor: "transparent",
            borderColor: "#334155",
            color: "#334155 !important"
          }
        },
        contained: {
          boxShadow: "none",
          "&:hover": {
            boxShadow: "none",
            backgroundColor: "transparent",
            color: "#334155"
          }
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
        inputSizeSmall: {
          paddingTop: 8,
          paddingBottom: 8
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
