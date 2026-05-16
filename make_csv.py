import pandas as pd
import numpy as np

def generate_mock_csv(path="mock_user.csv", n=1500):
    np.random.seed(0)
    data = {
        "W_um": np.round(np.random.uniform(1,5,n),2),        # μm
        "CL_fF": np.round(np.random.uniform(5,20,n),2),      # fF
        "VDD_V": np.round(np.random.uniform(1,1.5,n),2),     # V
        "stages": np.random.randint(3,10,n),                 # stage
        "delay_ps": np.round(np.random.uniform(10,30,n),2)   # 实测延迟 ps
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"模拟 CSV 已生成: {path}")

if __name__=="__main__":
    generate_mock_csv()
