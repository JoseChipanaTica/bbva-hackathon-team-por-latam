import { Snackbar } from "@mui/material";
import { customPalette } from "../../../theme";

export const PopupMessage = ({ open, handleClose, message }: any) => {
  return (
    <Snackbar
      sx={{
        mt: 5,
        "& .MuiPaper-root": {
          backgroundColor: customPalette.paperDarkHalfSolid,
        },
      }}
      anchorOrigin={{ vertical: "top", horizontal: "right" }}
      open={open}
      onClose={handleClose}
      message={message}
      autoHideDuration={5000}
      key={"topright"}
    />
  );
};
