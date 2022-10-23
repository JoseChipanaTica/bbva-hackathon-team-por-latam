import { PanTool } from "@mui/icons-material";
import { Divider, Typography, Box, Button } from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { SideDrawerModal } from "../components/side-drawer-modal/side-drawer-modal";
import { ClientForm } from "./client-form";

export const EditClientPage = () => {
  const [open, setOpen] = useState(true);
  const navigate = useNavigate();
  const toggleDrawer =
    (open: boolean) => (event: React.KeyboardEvent | React.MouseEvent) => {
      if (
        event.type === "keydown" &&
        ((event as React.KeyboardEvent).key === "Tab" ||
          (event as React.KeyboardEvent).key === "Shift")
      ) {
        return;
      }
      navigate(-1);
      //   setOpen(open);
    };
  return (
    <SideDrawerModal toggleDrawer={toggleDrawer} open={open} width={350}>
      <Typography variant="body1">Modificar información del cliente</Typography>
      <Box sx={{ py: 1 }} />
      <Divider />
      <Box sx={{ py: 4 }} />
      <ClientForm />
      <Box sx={{ py: 4 }} />
      <Box
        sx={{
          height: "30vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-evenly",
          alignItems: "center",
        }}
      >
        <Button variant="contained">
          <PanTool />
        </Button>
        <Button variant="contained">Modificar información</Button>
      </Box>
    </SideDrawerModal>
  );
};
