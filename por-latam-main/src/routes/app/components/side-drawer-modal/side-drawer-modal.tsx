import * as React from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import { customPalette } from "../../../../theme";

export const SideDrawerModal = (props: any) => {
  const { toggleDrawer, open, width, children } = props;
  return (
    <div>
      <Drawer
        anchor={"right"}
        sx={{
          position: "relative",
          zIndex: 10000,
          "& .MuiDrawer-paper": {
            backgroundColor: customPalette.paperDarkHalfSolid,
          },
        }}
        open={open}
        onClose={toggleDrawer(false)}
      >
        <Box
          sx={{ width, p: 2 }}
          role="presentation"
          //   onClick={toggleDrawer(false)}
          //   onKeyDown={toggleDrawer(false)}
        >
          {children}
        </Box>
      </Drawer>
    </div>
  );
};

// export const SideDrawerModal = () => {

// }
