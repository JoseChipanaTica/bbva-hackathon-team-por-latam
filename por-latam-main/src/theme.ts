import { createTheme } from "@mui/material";

export const customPalette = {
  primary: "#F54283",
  appBgDark: "#313149",
  paperDarkTransparent: "rgba(5,5,40,0.3)",
  paperDarkHalfSolid: "rgba(5,5,40,0.7)",
};

export const theme = createTheme({
  components: {
    MuiListItemIcon: {
      styleOverrides: {
        root: {
          color: customPalette.primary,
        },
      },
    },
    MuiContainer: {
      styleOverrides: {
        root: {
          backgroundColor: "transparent",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: customPalette.paperDarkTransparent,
        },
      },
    },
    MuiBackdrop: {
      styleOverrides: {
        root: {
          backdropFilter: "blur(5px)",
        },
      },
    },
    // MuiTable: {
    //   styleOverrides: {
    //     root: {
    //       webkitScro: {
    //         width:
    //       }
    //     },
    //   },
    // },
  },
  palette: {
    mode: "dark",
    primary: {
      main: customPalette.primary,
    },
  },
});
