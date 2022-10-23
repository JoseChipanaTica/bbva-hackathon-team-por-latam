import { CheckCircle, PanToolOutlined } from "@mui/icons-material";
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  IconButton,
  Typography,
  Stepper,
  StepButton,
  StepLabel,
  Step,
  StepContent,
  Paper,
  CircularProgress,
  Backdrop,
} from "@mui/material";
import axios from "axios";
import { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { customPalette } from "../../../theme";
import { PopupMessage } from "../components/popup-message";

const SERVICE_ENDPOINT = "54.204.100.30/login";

/* 

hand_left_palmer
hand_right_palmer
hand_right_dorsal
hand_left_dorsal
*/

interface HandCaptures {
  palmLeft: string;
  backLeft: string;
  palmRight: string;
  backRight: string;
}
const steps: {
  label: string;
  description: string;
  name: keyof HandCaptures;
}[] = [
  {
    label: "Palma izquierda",
    description: `Coloca la palma de la mano izquierda en la cámara.`,
    name: "palmLeft",
  },
  {
    label: "Dorsal izquierda",
    description: `Gira la mano izquierda y toma una foto del dorso de la mano.`,
    name: "backLeft",
  },
  {
    label: "Palma derecha",
    description: `Ahora coloca la palma de la mano derecha`,
    name: "palmRight",
  },
  {
    label: "Dorsal derecha",
    description: `Por último, gira la mano derecha y toma una foto del dorso de la mano.`,
    name: "backRight",
  },
];
export const AuthenticatorPage = () => {
  const [handCaptures, setHandCaptures] = useState<HandCaptures>({
    palmLeft: "",
    backLeft: "",
    palmRight: "",
    backRight: "",
  });
  const [open, setOpen] = useState(false);
  const [messageOpen, setMessageOpen] = useState(false);
  const handleClose = () => {
    setOpen(false);
  };
  const handleToggle = () => {
    setOpen(!open);
  };

  const fakeRequest = () => {
    setOpen(true);
    setTimeout(() => {
      setOpen(false);
      setMessageOpen(true);
    }, 2000);
  };
  const [activeStep, setActiveStep] = useState(0);
  //   const [imageStep, setImageStep] = useState<(HTMLImageElement | null)[]>([])

  async function onSumbitPhoto(file: any) {
    const formData = new FormData();
    // formData.append("hand_left_palmer", file);
    // formData.append("hand_right_palmer", file);
    // formData.append("hand_left_dorsal", file);
    formData.append("hand_right_dorsal", file);
    console.log(file);
    setOpen(true);
    try {
      const response = await axios({
        method: "post",
        url: "http://" + SERVICE_ENDPOINT,
        data: formData,
        headers: {
          "Content-Type": "multipart/form-data",
          "Access-Control-Allow-Origin": "*",
        },
      });

      console.log(response);
    } catch (error) {
      console.log(error);
      setOpen(false);
      return;
    }
    setOpen(false);
    setMessageOpen(true);
  }

  const webcamRef = useRef<any>(null);
  const captureHand = () => {
    if (!steps[activeStep]) return;
    try {
      const imageSrc = webcamRef.current.getScreenshot();
      setHandCaptures({ ...handCaptures, [steps[activeStep].name]: imageSrc });
    } catch (error) {
      console.log(error);
    }
  };

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setHandCaptures({
      palmLeft: "",
      backLeft: "",
      palmRight: "",
      backRight: "",
    });
  };
  const isAuthReady = () => {
    return Object.keys(handCaptures).some(
      (k) => handCaptures[k as keyof HandCaptures]
    );
  };
  const authStepper = (
    <Box sx={{ maxWidth: 400 }}>
      <Stepper nonLinear activeStep={activeStep} orientation="vertical">
        {steps.map((step, index) => (
          <Step key={step.label} completed={!!handCaptures[step.name]}>
            <StepButton onClick={() => setActiveStep(index)}>
              <StepLabel
                optional={
                  index === 2 ? (
                    <Typography variant="caption">Last step</Typography>
                  ) : null
                }
              >
                {step.label}
              </StepLabel>
            </StepButton>
            <StepContent>
              <Typography>{step.description}</Typography>
              <Box sx={{ mb: 2 }}>
                <div>
                  <Button
                    variant="contained"
                    onClick={handleNext}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    {index === steps.length - 1 ? "Finalizar" : "Continuar"}
                  </Button>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    Atrás
                  </Button>
                </div>
              </Box>
            </StepContent>
          </Step>
        ))}
      </Stepper>
      {activeStep === steps.length && (
        <Paper square elevation={0} sx={{ p: 3 }}>
          {isAuthReady() ? (
            <Typography variant="body1">Todo Listo</Typography>
          ) : (
            <Typography variant="body1">
              Tienes que realizar almenos una captura
            </Typography>
          )}
          <Button onClick={handleReset} sx={{ mt: 1, mr: 1 }}>
            Reset
          </Button>
        </Paper>
      )}
    </Box>
  );
  const miniatures = Object.keys(handCaptures).map((k) => {
    let key = k as keyof HandCaptures;
    return (
      <Card sx={{ mx: 2 }} key={key}>
        <CardContent>
          {handCaptures[key] ? (
            <img
              style={{ width: "60px" }}
              src={handCaptures[key]}
              alt={"imagen " + key}
              // onClick={() => {
              //   const a = document.createElement("a");
              //   a.href = "data:image/png;base64," + handCaptures[key]; //Image Base64 Goes here

              //   a.click(); //Downloaded file
              //   a.remove();
              // }}
            />
          ) : (
            <IconButton disabled>
              <PanToolOutlined />
            </IconButton>
          )}
        </CardContent>
      </Card>
    );
  });
  return (
    <>
      <Grid container spacing={1}>
        <Grid item xs={12} md={6}>
          <Box sx={{ position: "relative" }}>
            <Webcam
              ref={webcamRef}
              screenshotFormat="image/png"
              style={{ borderRadius: "10px", width: "100%" }}
              mirrored
            />
            <IconButton
              onClick={captureHand}
              size="large"
              sx={{
                position: "absolute",
                bottom: 0,
                left: "50%",
                transform: "translate(-50%, -50%)",
                borderWidth: "2px",
                borderStyle: "solid",
                borderColor: customPalette.primary,
              }}
            >
              <PanToolOutlined sx={{ color: customPalette.primary }} />
            </IconButton>
          </Box>
          <Box sx={{ p: 2, display: "flex", justifyContent: "center" }}>
            {miniatures}
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          {authStepper}
          <Button
            onClick={fakeRequest}
            disabled={!isAuthReady()}
            variant="outlined"
          >
            Autenticar
          </Button>
          {/* <Typography variant="body1">Stepper</Typography> */}
        </Grid>
        <FileUploadPage onSubmitPhoto={onSumbitPhoto} />
      </Grid>
      <CustomBackdrop
        open={open}
        handleToggle={handleToggle}
        handleClos={handleClose}
      />
      <PopupMessage
        open={messageOpen}
        handleClose={() => {
          setMessageOpen(false);
        }}
        message={
          <Box sx={{ display: "flex" }}>
            <CheckCircle color={"success"} />
            <Typography sx={{ mx: 2 }} color="white" variant="body1">
              Autencicación exitosa!
            </Typography>
          </Box>
        }
      ></PopupMessage>
    </>
  );
};

function FileUploadPage({ onSubmitPhoto }: any) {
  const [selectedFile, setSelectedFile] = useState<any>();
  const [isFilePicked, setIsFilePicked] = useState(false);

  const changeHandler = (event: any) => {
    setSelectedFile(event.target.files[0]);
    setIsFilePicked(true);
  };

  // const handleSubmission = () => {};

  return (
    <div>
      <input
        type="file"
        name="file"
        onChange={changeHandler}
        placeholder={"selecciona un archivo"}
      />
      <div>
        <Button
          disabled={!isFilePicked}
          onClick={() => {
            if (isFilePicked) {
              onSubmitPhoto(selectedFile);
            }
          }}
        >
          Enviar
        </Button>
      </div>
    </div>
  );
}

// const [open, setOpen] = useState(false);
// const handleClose = () => {
//   setOpen(false);
// };
// const handleToggle = () => {
//   setOpen(!open);
// };

export function CustomBackdrop({ handleClose, handleToggle, open }: any) {
  return (
    <div>
      <Backdrop
        sx={{ color: "#fff", zIndex: (theme: any) => theme.zIndex.drawer + 1 }}
        open={open}
        onClick={handleClose}
      >
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <CircularProgress color="inherit" />
          <Typography sx={{ fontSize: "24px", pt: 2 }}>
            Estamos procensando su solicitud
          </Typography>
        </Box>
      </Backdrop>
    </div>
  );
}
