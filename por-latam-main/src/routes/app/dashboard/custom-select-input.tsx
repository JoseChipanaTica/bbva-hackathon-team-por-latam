import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormHelperText from "@mui/material/FormHelperText";
import FormControl from "@mui/material/FormControl";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import { useState } from "react";

export const CustomSelectInput = () => {
  const [option, setOption] = useState("1");

  const handleChange = (event: SelectChangeEvent) => {
    setOption(event.target.value);
  };

  return (
    <FormControl sx={{ m: 1, minWidth: 120 }}>
      <Select
        value={option}
        onChange={handleChange}
        displayEmpty
        inputProps={{ "aria-label": "Without label" }}
        size="small"
      >
        <MenuItem value="1">Opción 1</MenuItem>
        <MenuItem value="2">Opción 2</MenuItem>
        <MenuItem value="3">Opción 3</MenuItem>
        <MenuItem value="4">Opción 4</MenuItem>
      </Select>
      <FormHelperText>Selecciona una opción</FormHelperText>
    </FormControl>
  );
};
