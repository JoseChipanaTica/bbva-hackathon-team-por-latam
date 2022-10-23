import { Container, Box, Typography } from "@mui/material";

export const SignupPage = () => {
  return (
    <Container maxWidth="xl">
      <Box
        sx={{
          backgroundColor: "yellow",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="h1">Signup</Typography>
      </Box>
    </Container>
  );
};
