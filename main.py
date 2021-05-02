import pandas as pd
import gnsspy as gp
import georinex as gr

class DataWrapper():

    def __init__(self, file_url: str):
        station = gp.read_obsFile("data/S4106L.21o")
        # orbit = gp.sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="esa") # this does not work
        # spp_result = gp.spp(station, orbit, system='G', cut_off=7.0) 
        gpr_data = gr.load("data/S4106L.21o")
        print(gpr_data)

def main():
    data_wrapper: DataWrapper = DataWrapper('')
    pass

if __name__ == "__main__":
    main()