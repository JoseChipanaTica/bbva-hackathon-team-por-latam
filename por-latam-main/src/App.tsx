import { Box, ThemeProvider } from "@mui/material";
import { RouterProvider } from "react-router-dom";

import { router } from "./app-router";
import { theme } from "./theme";

import appImage from "./appBg.png";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          display: "flex",
          backgroundImage: `url(${appImage})`,
          backgroundSize: "100% 100%",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
      >
        {/* <img
        src={appImage}
        style={{ position: "absolute", width: "100vw", height: "100vh" }}
        alt="background"
        draggable={false}
      /> */}
        <Box
          sx={{
            flex: 1,
            // position: "absolute",
            backgroundColor: "rgba(49,49,73,0.9)",
            height: "100vh",
            // width: "100vw",
          }}
        >
          <RouterProvider router={router} />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
