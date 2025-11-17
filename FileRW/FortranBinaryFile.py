import os, sys 
import numpy as np
import struct

sys.path.append(os.path.dirname("Utilities"))
from Utilities.DataType import DataType

class FortranBinaryFile:
    def __init__(self, filename=None, filepath=None, overwrite=False):
        self.f = None
        self.filename  = filename
        self.filepath  = filepath
        self.overwrite = overwrite
        self.current_offset = 0
        #if filename:
        #    self.set_filename(filename, filepath, overwrite)

    @property
    def eof(self):
        if self.f:
            return self.f.tell() == os.fstat(self.f.fileno()).st_size
        else:
            return False

    def open(self):
        mode = "wb"

        if self.filepath:
            if os.path.exists(f"{self.filepath}{self.filename}"):
                mode = "rb"
            if self.overwrite:
                mode = "wb"
            self.f = open(f"{self.filepath}{self.filename}", mode)
        else:
            if os.path.exists(f"{self.filename}"):
                mode = "rb"
            if self.overwrite:
                mode = "wb"
            self.f = open(f"{self.filename}", mode)
    
    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
        
    def __del__(self):
        if self.f:
            self.f.close()
            self.f = None

    def write_string(self, str_data, str_len):
        """
            When writing strings, theyre length needs to be specified. Fortran strings are 
            initialised at certain lengths and that much data is written, regardless of 
            how long the final string is. Also need to include headers (data block size)
        """
        self.write_int(str_len)
        str_data += " "*(str_len - len(str_data))
        self.f.write(bytearray(str_data, "utf-8"))
        self.write_int(str_len)

    def read_string(self) -> tuple:
        str_len = self.read_int()
        #print(str_len)
        s = self.f.read(str_len).decode('utf-8')
        str_end = self.read_int()
        assert str_len == str_end
        return s, str_len

    def write_int(self, num, python_indexed=False, negate=False):
        """
            python_indexed: fortran is 1 indexed, python is 0 indexed. Flag 
                            usually passed down by write_record to handle 
                            converting large matrices in one go. 
            negate        : not sure why we need this. maybe for writing sub records?
            
        """
        if python_indexed:
            num += 1
        if negate:
            num = -num
        self.f.write(struct.pack('<i', num)) # <i = little-endian 4 byte integer

    def read_int(self) -> int:
        return int(struct.unpack('<i', self.f.read(4))[0])

    def write_real(self, num, sz='d'):
        """
            TODO - expand this to handle floats?
        """
        self.f.write(struct.pack(sz, num))

    def read_real(self) -> float:
        pass

    def write_record(self, matrix, dt=DataType.INTEGER, python_indexed=False) -> None:
        """
            Params:
                matrix         - NxM matrix. write_record will convert to np.array
                dt             - DataType from enumeration in FortranBinaryFile 
                python_indexed - True if matrix is 0 indexed. will 1 index all values.
            Return:
                None
        """
        if type(matrix) != np.array:
            matrix = np.array(matrix)

        # convert into fortran style column major array
        #matrix = matrix.flatten('F') ## flatten may work better here?
        matrix = np.hstack(matrix)

        # calculate size of record in bytes
        byte_count = 4 * len(matrix)
        if dt==DataType.DOUBLE: byte_count *= 2

        # write record prefix/header
        self.write_int(byte_count)

        # write record elements
        if dt==DataType.DOUBLE:        
            for x in matrix:
                self.write_real(x, sz='d') # why dont we need to specify endianess here?
        elif dt==DataType.FLOAT:        
            for x in matrix:
                self.write_real(x, sz='<f') # why dont we need to specify endianess here?
        elif dt==DataType.INTEGER:  
            for x in matrix:
                self.write_int(x, python_indexed)
        
        # write record suffix/tail
        self.write_int(byte_count)

    def read_data(self, init_offset, dt=DataType.INTEGER):
        if dt == DataType.INTEGER:
            dformat = '<i' 
            dtype = np.int32
        elif dt == DataType.DOUBLE:
            dformat = '<d'
            dtype = np.double
        elif dt == DataType.FLOAT:
            dformat = '<f'
            dtype = np.float32
        
        # read data
        pos = init_offset
        data = []
        self.f.seek(pos)
        prefix = struct.unpack( '<i', self.f.read(4))[0]
        pos += 4
        if prefix < 0:
            suffix = 1
            while suffix > 0:
                pos += abs(prefix)
                #self.f.seek(pos)
                #print(f"data = {len(data)}, type = {type(data)}")
                data += self.f.read(abs(prefix)) # replace with array data read
                #print(f"data = {len(data)}, type = {type(data)}")
                suffix = struct.unpack( '<i', self.f.read(4))[0]
                pos += 4
                if suffix > 0:
                    prefix = struct.unpack( '<i', self.f.read(4))[0]
        else:
            pos += prefix
            #self.f.seek(pos)
            data = self.f.read(prefix) # replace with array data read
            suffix = struct.unpack( '<i', self.f.read(4))[0]
            pos += 4
        # end read data
        # parse data
        #print(f"data = {len(data)}, type = {type(data)}")
        data_parsed = np.array([x[0] for x in struct.iter_unpack(dformat, data)])
        #print(f"read - {out_array.shape}")
        return data_parsed, pos
    
    def read_record(self, init_offset, output_size, dt=DataType.INTEGER):
        d, _ = self.read_data(init_offset, dt)
        out_array = np.reshape(d, output_size)
        return out_array

    def read_record_auto(self, output_size, dt=DataType.INTEGER):
        init_offset = self.f.tell()
        prefix = struct.unpack('<i', self.f.read(4))[0]
        end_offset = init_offset + 4 + prefix + 4
        return self.read_record(init_offset, output_size, dt)

    def read_record_data(self, step=4, output_size=None):
        init_offset = self.current_offset
        prefix = struct.unpack('<i', self.f.read(4))[0]
        end_offset = init_offset + 4 + prefix + 4
        data, end_offset = self.read_data(init_offset)
        self.current_offset = end_offset
        if output_size is not None:
            data = np.reshape(data, output_size)
        print(f"data ({len(data)}) = {data}")
        if len(data) == 1:
            data = data[0]
        return data

    @staticmethod
    def calc_offset(base, elem_num, nodes, data_size):
        byte_count = elem_num * nodes * data_size
        offset = 0
        if byte_count < 2147483639:
            offset = 8 + base + byte_count
            return offset
        counter = byte_count
        i = 1
        while counter - 2147483639 > 0:
            offset += 8
            offset += 2147483639
            counter -= 2147483639
            i += 1
            
        offset += 8
        offset += byte_count % 2147483639
        offset += base
        return offset
    