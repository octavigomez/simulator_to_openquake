# Description

This repository provides a set of Python tools to convert earthquake cycle simulator outputs into OpenQuake compatible source model formats for probabilistic seismic hazard calculations.
Simulated fault ruptures are treated as gridded ruptures, with each fault mesh element center being a node of the grid. Each rupture is considered as a characteristic source according to the OpenQuake nomenclature and its rate is the inverse
of the simulated catalogue length. 

## Key Features

🧩 Simulator Integration – Reads and processes rupture catalogs from numerical earthquake simulators.

🔄 Format Conversion – Translates rupture data into OpenQuake XML format.

⚙️ Customizable Workflow – Modular scripts that can be adapted to different simulators and fault geometries.

## Codes and structure

*/lib/parser_XML.py* --> Contains parse and xml conversion functions that translate earthquake simulator ruptures into an OpenQuake XML.

*extract_ruptures_to_XML.py* --> Main Python script for rupture extraction and conversion from own simulated catalogues. Currently the code supports only RSQSim outputs.

## Code execution

1. Clone the repository and install the necessary requirements from the requirements.txt file in your working venv.
   
2. In the code *extract_ruptures_to_XML.py* change *path_in* variable to the directory where you have your simulated catalogue(s). Each catalogues should have its own directory within the main path. The code is designed to sequentially read and extract ruptures from several simulated catalogue directories. 

3. Inside the simulated catalogue directory should be a) the simulated catalogue in *json* format and a fault model in *csv*, delimited by *;* an following the RSQSim input file format.

4. Adapt global variables at your convenience:
   - *m_filtering* defines the minimum magnitude for rupture extraction
   - *cut_year* defines the minimum simulation year (in catalogue years)
   - *time_windows* defines the length of the catalogue you want to extract ruptures from (in catalogue years)
   - *patch_threshold* allows you to discard earthquake ruptures that involve less than *n* number of patches. Default is set at 0.
     
5. Run the code. Output XML OpenQuake source model will be saved in the same path as the *path_in* variable you set at step 2.

## License

This repository runs under a Creative Commons License CC-BY-NC 4.0. More info at: https://creativecommons.org/licenses/by-nc/4.0/deed
