# FlyBrainProject
This is the repository for the project titled "A computational model of the integration of landmarks and motion in the insect central complex"

## Instructions on how to run the model
1. Clone this repository
2. Go to Project_as5624_pb2735.ipynb and run the cells that test the AVDU and Landmark features and create the file input.h5
3. In the server, go to ~/ffbo/neurodriver/examples/ring
4. Exp. 1: For testing positional inputs, set rota, rotb weights to 0. and set pos weights to 0.1 in RingAttractorNetwork.py
5. Run:
  * python RingAttractorNetwork.py
  * python demo.py
6. Use sftp to get the file output.h5. Use the corresponding cell in the Jupyter Notebook to visualize the output. Remember to backup all the output.h5 files after each experiment.
7. Exp. 2: For testing driver (AVDU) inputs, set pos weights to 0. in RingAttractorNetwork.py, rota, rotb weights to 0.1. Then, repeat steps 5-6.
8. Exp. 3: For testing integration of landmark and motion, set pos weights to 0.01 in RingAttractorNetwork.py. Then, repeat steps 5-6.
9. To see the estimated and true azimuth, run the calculate_azimuth() function in the Jupyter Notebook and the two cells following that. Change the filenames passed to calculate_azimuth() to where you have stored the output.h5 backups (we recommend doing it as is shown in the Jupyter Notebook)