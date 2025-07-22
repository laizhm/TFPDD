## Implementation Notes

The watermark embedding and extraction processes in this project are implemented based on MATLAB. 

### Prerequisites:
1. **MATLAB Runtime (MCR) Requirement**  
   You need to download the MATLAB Runtime (MCR) from:  
   https://ww2.mathworks.cn/products/compiler/matlab-runtime.html  
   Unzip the downloaded package in the current project directory.

   ⚠️ **Important Path Note**:  
   The code references MCR at: `./MCR/v93/`

2. **Python Dependencies**  
   Install required Python packages:  
   ```bash
   pip install -r requirements.txt
   ```

### Execution:
- **Watermark Embedding**  
  ```bash
  python embeding.py --inp './test_img/' --outp './embed/' --l 32
  ```

- **Watermark Extraction**  
  ```bash
  python ber.py --inp './embed/' --outp './result/' --l 32
  ```
```