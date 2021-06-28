
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

void readArray(fstream& in_file, int64_t num_elems, double* arr) {
  for(int i=0; i<num_elems; i++) {
    double num;
    in_file >> num;
    arr[i] = num;
  }
}

void writeMat2Fut(fstream& out_file, int64_t num_rows, int64_t num_cols, double* mat) {
  out_file << "[\n";
  for(int i=0; i<num_rows; i++) {
    out_file << "  [ ";
    if (num_cols > 0) {
      out_file << std::setprecision(16) << mat[i*num_cols] << "f64";
    }
    for(int j=1; j<num_cols; j++) {
      out_file << ", ";
      out_file << std::setprecision(16) << mat[i*num_cols + j] << "f64";
    }
    out_file << " ]";
    if(i < num_rows-1)
      out_file << ",";
    out_file << "\n";
  }
  out_file << "]\n\n";
}


int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <ADbench-data-file.txt> <Futhark-data-file.txt> " << std::endl;
    return 1;
  }

  const char *input_file_name = argv[1];
  const char *output_file_name = argv[2];

  int64_t slen, d, seq_len;
  double *mainParams, *extraParams, *state, *inpSeq;

  fstream in_file;
  in_file.open(input_file_name, ios::in);
  if (!in_file) {
    cout << "Input file not found, Exiting!" << endl;
    exit(1);
  }

  fstream out_file;
  out_file.open(output_file_name, ios::out);
  if (!out_file) {
    cout << "Could not create output file, Exiting!" << endl;
    exit(2);
  }

  // start reading:
  in_file >> slen;
  in_file >> seq_len;
  in_file >> d;

  printf("slen: %ld, d: %ld, seq_len: %ld\n\n", slen, d, seq_len);

  mainParams = new double[2*slen*4*d];
  extraParams= new double[3*d];
  state = new double[2*slen*d];
  inpSeq = new double[seq_len*d];

  readArray(in_file, (2*slen)*4*d, mainParams);
  readArray(in_file, 3*d, extraParams);
  readArray(in_file, 2*slen*d, state);
  readArray(in_file, seq_len*d, inpSeq);

  in_file.close();

  writeMat2Fut(out_file, 2*slen, 4*d, mainParams);
  writeMat2Fut(out_file, 3, d, extraParams);
  writeMat2Fut(out_file, 2*slen, d, state);
  writeMat2Fut(out_file, seq_len, d, inpSeq);

  out_file.close();
  delete[] mainParams;
  delete[] extraParams;
  delete[] state;
  delete[] inpSeq;
}