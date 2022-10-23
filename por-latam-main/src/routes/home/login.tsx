import {
  EmailOutlined,
  Lock,
  Visibility,
  VisibilityOff,
} from "@mui/icons-material";
import {
  Box,
  Typography,
  Grid,
  TextField,
  Button,
  InputAdornment,
  FormControl,
  OutlinedInput,
  IconButton,
  InputLabel,
} from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { customPalette } from "../../theme";
import loginImage from "./login/login-img.png";
interface LoginState {
  email: string;
  password: string;
  showPassword: boolean;
}

export const LoginPage = () => {
  const [values, setValues] = useState({
    email: "",
    password: "",
    showPassword: false,
  });
  const navigate = useNavigate();
  const handleChange =
    (prop: keyof LoginState) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setValues({ ...values, [prop]: event.target.value });
    };
  const handleClickShowPassword = () => {
    setValues({
      ...values,
      showPassword: !values.showPassword,
    });
  };
  const isDisabled = () => {
    return !(values.email && values.password);
  };
  return (
    <Grid container>
      <Grid item xs={12} md={6} lg={8}>
        <Grid
          container
          spacing={3}
          sx={{ margin: "auto", mt: "24%", px: "25%" }}
        >
          <Grid item xs={12}>
            <Typography sx={{ fontSize: "34px", color: customPalette.primary }}>
              Bienvenido
            </Typography>
            <Typography sx={{ color: "white" }} variant="body1">
              Ingrese sus credenciales de administrador.
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <TextField
              id="input-with-icon-textfield"
              label="Email"
              value={values.email}
              onChange={handleChange("email")}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <EmailOutlined />
                  </InputAdornment>
                ),
              }}
              fullWidth
              variant="outlined"
            />
            {/* <TextField fullWidth label="Email" /> */}
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth variant="outlined">
              <InputLabel htmlFor="outlined-adornment-password">
                Password
              </InputLabel>
              <OutlinedInput
                id="outlined-adornment-password"
                type={values.showPassword ? "text" : "password"}
                value={values.password}
                onChange={handleChange("password")}
                startAdornment={
                  <InputAdornment position="start">
                    <Lock />
                  </InputAdornment>
                }
                endAdornment={
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={handleClickShowPassword}
                      onMouseDown={(e) => e.preventDefault()}
                      edge="end"
                    >
                      {values.showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                }
                label="Password"
              />
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <Button
              onClick={() => navigate("app")}
              disabled={isDisabled()}
              variant="contained"
            >
              Iniciar Sesión
            </Button>
          </Grid>
          <Grid item xs={12}>
            <Typography color="white" variant="caption">
              ¿No tienes cuenta?{" "}
              <Typography color="inherit" variant="body2">
                Regístrate
              </Typography>
            </Typography>
          </Grid>
        </Grid>

        {/*  */}
      </Grid>
      <Grid item xs={12} md={6} lg={4}>
        <Box
          sx={{
            height: "100vh",
            backgroundImage: `url(${loginImage})`,
            backgroundRepeat: "no-repeat",
            backgroundPosition: "center",
            backgroundSize: "cover",
          }}
        />
        {/* <Container>
          <Typography> </Typography>
        </Container> */}
      </Grid>
    </Grid>
  );
};
