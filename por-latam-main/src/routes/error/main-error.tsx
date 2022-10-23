import { useRouteError } from "react-router-dom";
import { Container, Box, Typography } from "@mui/material";

export const MainErrorPage = () => {
  const error = useRouteError() as any;
  return (
    <Container maxWidth="md">
      <Box
        sx={{
          backgroundColor: "green",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="h1">Oops!</Typography>
        <Typography variant="body2">
          Sorry, an unexpected error has ocurred.
        </Typography>
        <Typography variant="subtitle2">
          {error.statusText || error.message}
        </Typography>
      </Box>
    </Container>
  );
};
