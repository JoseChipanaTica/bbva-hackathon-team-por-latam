import {
  Divider,
  Drawer as MuiDrawer,
  styled,
  Toolbar,
  IconButton,
} from "@mui/material";

import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";

import { APP_DRAWER_WIDTH } from "../../../constants";
import { ListItems } from "./list-items";

const Drawer = styled(MuiDrawer, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
  "& .MuiDrawer-paper": {
    position: "relative",
    whiteSpace: "nowrap",
    width: APP_DRAWER_WIDTH,
    transition: theme.transitions.create("width", {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
    boxSizing: "border-box",
    ...(!open && {
      overflowX: "hidden",
      transition: theme.transitions.create("width", {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
      }),
      width: theme.spacing(7),
      [theme.breakpoints.up("sm")]: {
        width: theme.spacing(9),
      },
    }),
  },
}));

interface AppDrawerProps {
  open: boolean;
  toggleDrawer: () => void;
}

export const AppDrawer = ({ open, toggleDrawer }: AppDrawerProps) => {
  return (
    <Drawer variant="permanent" open={open}>
      <Toolbar
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          px: [1],
        }}
      >
        <IconButton onClick={toggleDrawer} sx={{ opacity: open ? 1 : 0 }}>
          <ChevronLeftIcon />
        </IconButton>
      </Toolbar>
      <Divider />
      <ListItems />
    </Drawer>
  );
};

/* 
<List component="nav">
        {mainListItems}
        <Divider sx={{ my: 1 }} />
        {secondaryListItems}
      </List>
*/
