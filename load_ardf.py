import numpy as np
import struct
import zlib
import os
# import h5py # PN
import gc

def debug_print(dstr):
    if not dstr.startswith("Info"):
        print(dstr)


def ARDF_get_next_element(fd):
    try:
        header = fd.read(0x8)
    except:
        debug_print("Error: ARDF_get_next_element - cannot read header!")
        return -1

    crc32, size = ARDF_parse_header(header)

    data_size = size - 0x8

    if data_size <= 0:
        debug_print("Error: ARDF_get_next_element - not a valid element! (assert size > 0)")
        return -1

    try:
        data = fd.read(size - 0x8)
    except:
        debug_print("Error: ARDF_get_next_element - cannot read data!")
        return -1

    if ARDF_check_header(header, data, crc32) != 0:
        debug_print("Error: ARDF_get_next_element - Error validating header CRC32!")
        return -1

    return [size, data]


def ARDF_parse_header(header):
    try:
        crc32 = struct.unpack("<L", header[0:4])[0]
        size = struct.unpack("<L", header[4:8])[0]
        debug_print("Info: ARDF_parse_header - CRC32=" + str(crc32) + " SIZE=" + str(size))
    except:

        # debug_print("Error: ARDF_parse_header - cannot parse!") # PN commented
        # print(header)     # PN commented
        # return -1         # PN commented
        crc32 = ''          # PN added
        size = ''           # PN added

    return [crc32, size]


def ARDF_check_header(header, data, crc32):
    if zlib.crc32(header[4:8] + data) % (1 << 32) != crc32:
        debug_print("Error: ARDF_check_header - header CRC32 does not match!")
        return -1

    return 0


def ARDF_get_FTOC_entry(fd, ftoc_entry_size):
    try:
        header = fd.read(ftoc_entry_size)
    except:
        debug_print("Error: ARDF_get_FTOC_entry - cannot read header!")
        return -1

    crc32, size = ARDF_parse_header(header)

    if crc32 == 0 or size == 0:
        debug_print("Info: ARDF_get_FTOC_entry - Not a valid entry, maybe empty...")
        return -1

    if zlib.crc32(header[4:]) % (1 << 32) != crc32:
        debug_print("Error: ARDF_get_FTOC_entry - header CRC32 does not match!")
        return -1

    crc32 = struct.unpack("<L", header[0:4])[0]
    size = struct.unpack("<L", header[4:8])[0]
    magic = struct.unpack("<4s", header[8:12])[0].decode("ascii")
    unk1 = struct.unpack("<L", header[12:16])[0]
    offset = struct.unpack("<L", header[16:20])[0]
    unk2 = struct.unpack("<L", header[20:24])[0]

    return [magic, offset, size, [crc32, unk1, unk2]]


def ARDF_get_TTOC_entry(fd, ttoc_entry_size):
    try:
        header = fd.read(ttoc_entry_size)
    except:
        debug_print("Error: ARDF_get_TTOC_entry - cannot read header!")
        return -1

    crc32, size = ARDF_parse_header(header)

    if crc32 == 0 or size == 0:
        debug_print("Info: ARDF_get_TTOC_entry - Not a valid entry, maybe empty...")
        return -1

    if zlib.crc32(header[4:]) % (1 << 32) != crc32:
        debug_print("Error: ARDF_get_TTOC_entry - header CRC32 does not match!")
        return -1

    crc32 = struct.unpack("<L", header[0:4])[0]
    size = struct.unpack("<L", header[4:8])[0]
    magic = struct.unpack("<4s", header[8:12])[0].decode("ascii")
    unk1 = struct.unpack("<L", header[12:16])[0]
    unk2 = struct.unpack("<L", header[16:20])[0]
    unk3 = struct.unpack("<L", header[20:24])[0]
    offset = struct.unpack("<L", header[24:28])[0]
    unk4 = struct.unpack("<L", header[28:32])[0]

    return [magic, offset, size, [crc32, unk1, unk2, unk3, unk4]]


def ARDF_parse_TOC(fd, toc_hdr_size, toc_hdr, toc_entry_func):
    magic_bytes = struct.unpack("<4s", toc_hdr[0:4])[0].decode("ascii")
    unk1 = struct.unpack("<L", toc_hdr[4:8])[0]
    toc_size = struct.unpack("<L", toc_hdr[8:12])[0]
    unk3 = struct.unpack("<L", toc_hdr[12:16])[0]
    toc_entry_count = struct.unpack("<L", toc_hdr[16:20])[0]
    toc_entry_size = struct.unpack("<L", toc_hdr[20:24])[0]

    toc_entries = []
    for i in range(toc_entry_count):
        try:
            entry = toc_entry_func(fd, toc_entry_size)
            if entry != -1:
                toc_entries.append(entry)
        except:
            pass
    return toc_entries


def ARDF_parse_FTOC(fd, ftoc_hdr_size, ftoc_hdr):
    return ARDF_parse_TOC(fd, ftoc_hdr_size, ftoc_hdr, ARDF_get_FTOC_entry)


def ARDF_parse_TTOC(fd, ttoc_hdr_size, ttoc_hdr):
    return ARDF_parse_TOC(fd, ttoc_hdr_size, ttoc_hdr, ARDF_get_TTOC_entry)


def ARDF_load_file(datafile):
    fd = open(datafile, "rb")

    fd.seek(0, os.SEEK_END)
    ardf_file_size = fd.tell()
    fd.seek(0)

    ardf_ftoc = []
    ardf_ttoc = []

    # Reading header
    size, data = ARDF_get_next_element(fd)

    try:
        magic_bytes = struct.unpack("<4s", data[0:4])[0].decode("ascii")
        unknown = struct.unpack("<L", data[4:8])[0]
    except:
        debug_print("Error: load_ardf - header cannot be parsed!")
        return -1

    if magic_bytes != 'ARDF':
        debug_print("Error: load_ardf - header magic bytes do not match!")
        return -1
    else:
        debug_print("Info: load_ardf - found ARDF header.")

    if unknown != 0:
        debug_print("Warning: load_ardf - header does not end with 0... Continuing but something might be wrong!")

    while True:
        try:
            if fd.tell() == ardf_file_size:
                debug_print("Info: File parsed successfully!")
                break
            size, data = ARDF_get_next_element(fd)

            magic_bytes = struct.unpack("<4s", data[0:4])[0].decode("ascii")

            if magic_bytes == 'FTOC':
                # debug_print("Warning: load_ardf - Found FTOC, parsing.")
                ardf_ftoc = ARDF_parse_FTOC(fd, size, data)
            elif magic_bytes == 'TTOC':
                # debug_print("Warning: load_ardf - Found TTOC, parsing.")
                ardf_ttoc = ARDF_parse_TTOC(fd, size, data)
            else:
                break

        except:
            if ardf_ftoc == [] or ardf_ttoc == []:
                debug_print("Error: Could not find FTOC and TTOC!")
            break
    return ardf_ftoc, ardf_ttoc, fd


def ARDF_get_TEXT_from_TOFF(fd, toff_entry):
    toff_entry_offset = toff_entry[1]
    fd.seek(toff_entry_offset)
    toff_entry = ARDF_get_next_element(fd)

    toff_entry_size = toff_entry[0]
    toff_entry_data = toff_entry[1]

    magic_bytes = struct.unpack("<4s", toff_entry_data[0:4])[0].decode("ascii")
    unk1 = struct.unpack("<L", toff_entry_data[4:8])[0]
    unk2 = struct.unpack("<L", toff_entry_data[8:12])[0]
    text_size = struct.unpack("<L", toff_entry_data[12:16])[0]
    text_data = toff_entry_data[16:16 + text_size]

    return text_data


def ARDF_get_VOLM(fd, volm_entry):
    volm_entry_offset = volm_entry[1]
    fd.seek(volm_entry_offset)
    volm_entry = ARDF_get_next_element(fd)
    volm_ftoc = ARDF_parse_FTOC(fd, volm_entry[0], volm_entry[1])
    volm_ttoc_entry = ARDF_get_next_element(fd)
    volm_ttoc = ARDF_parse_TTOC(fd, volm_ttoc_entry[0], volm_ttoc_entry[1])

    vdef_entry = ARDF_get_next_element(fd)[1]
    magic_bytes = struct.unpack("<4s", vdef_entry[0:4])[0].decode("ascii")
    unk1 = struct.unpack("<L", vdef_entry[4:8])[0]
    x_pixcount = struct.unpack("<L", vdef_entry[8:12])[0]
    y_pixcount = struct.unpack("<L", vdef_entry[12:16])[0]
    unk2 = struct.unpack("<d", vdef_entry[16:24])[0]
    unk3 = struct.unpack("<d", vdef_entry[24:32])[0]
    unk4 = struct.unpack("<d", vdef_entry[32:40])[0]
    x_step = struct.unpack("<d", vdef_entry[40:48])[0]
    y_step = struct.unpack("<d", vdef_entry[48:56])[0]
    z_step = struct.unpack("<d", vdef_entry[56:64])[0]
    x_unit = struct.unpack("<32s", vdef_entry[64:96])[0].decode("ascii")
    y_unit = struct.unpack("<32s", vdef_entry[96:128])[0].decode("ascii")
    z_unit = struct.unpack("<32s", vdef_entry[128:160])[0].decode("ascii")
    section_names = struct.unpack("<32s", vdef_entry[160:192])[0].decode("ascii")
    section_number = struct.unpack("<L", vdef_entry[192:196])[0]

    volm_vdef = [section_names, section_number, x_pixcount, y_pixcount, x_step, y_step, z_step, x_unit, y_unit, z_unit]

    x_channel_names = []
    data_channels_names = []
    data_channels_units = []
    voff_data = []
    vset_data = []
    vnam_data = []
    vdat_data = []
    xdat_data = []

    while True:
        try:
            element_data = ARDF_get_next_element(fd)[1]
        except:
            break
        magic_bytes = struct.unpack("<4s", element_data[0:4])[0].decode("ascii")

        if magic_bytes == 'VCHN':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            channame = struct.unpack("<32s", element_data[8:40])[0]
            chanunit = struct.unpack("<32s", element_data[40:72])[0]
            data_channels_names.append(channame)
            data_channels_units.append(chanunit)
        elif magic_bytes == 'XDEF':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            unk2 = struct.unpack("<L", element_data[8:12])[0]
            x_length = struct.unpack("<L", element_data[12:16])[0]
            x_channel_names = struct.unpack("<" + str(int(x_length)) + "s", element_data[16:16 + int(x_length)])[0]
        elif magic_bytes == 'VTOC':
            pass
        elif magic_bytes == 'VOFF':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            unk2 = struct.unpack("<L", element_data[8:12])[0]
            unk3 = struct.unpack("<L", element_data[12:16])[0]
            unk4 = struct.unpack("<L", element_data[16:20])[0]
            unk5 = struct.unpack("<L", element_data[20:24])[0]
            unk6 = struct.unpack("<L", element_data[24:28])[0]
            voff_data.append([unk1, unk2, unk3, unk4, unk5, unk6])
        elif magic_bytes == 'VSET':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            num_point = struct.unpack("<L", element_data[8:12])[0]
            y_coord = struct.unpack("<L", element_data[12:16])[0]
            x_coord = struct.unpack("<L", element_data[16:20])[0]
            unk5 = struct.unpack("<L", element_data[20:24])[0]
            prev_offset = struct.unpack("<L", element_data[24:28])[0]
            next_offset = struct.unpack("<L", element_data[32:36])[0]
            unk8 = struct.unpack("<L", element_data[36:40])[0]
            vset_data.append([num_point, y_coord, x_coord, unk5, prev_offset, next_offset, unk8])
        elif magic_bytes == 'VNAM':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            num_point = struct.unpack("<L", element_data[8:12])[0]
            y_coord = struct.unpack("<L", element_data[12:16])[0]
            x_coord = struct.unpack("<L", element_data[16:20])[0]
            name_length = struct.unpack("<L", element_data[20:24])[0]
            meas_name = struct.unpack("<" + str(int(name_length)) + "s", element_data[24:24 + int(name_length)])[0]
            vnam_data.append([num_point, y_coord, x_coord, name_length, meas_name])
        elif magic_bytes == 'VDAT':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            num_point = struct.unpack("<L", element_data[8:12])[0]
            y_coord = struct.unpack("<L", element_data[12:16])[0]
            x_coord = struct.unpack("<L", element_data[16:20])[0]
            data_length = struct.unpack("<L", element_data[20:24])[0]
            channel_index = struct.unpack("<L", element_data[24:28])[0]
            cut1 = struct.unpack("<L", element_data[32:36])[0]
            cut2 = struct.unpack("<L", element_data[36:40])[0]
            cut3 = struct.unpack("<L", element_data[40:44])[0]
            cut4 = struct.unpack("<L", element_data[44:48])[0]
            meas_data = struct.unpack("<" + str(int(data_length)) + "f", element_data[48:48 + 4 * int(data_length)])
            vdat_data.append(
                [num_point, y_coord, x_coord, data_length, channel_index, cut1, cut2, cut3, cut4, meas_data])
        elif magic_bytes == 'XDAT':
            unk1 = struct.unpack("<L", element_data[4:8])[0]
            num_point = struct.unpack("<L", element_data[8:12])[0]
            y_coord = struct.unpack("<L", element_data[12:16])[0]
            x_coord = struct.unpack("<L", element_data[16:20])[0]
            num_xdata_points = struct.unpack("<L", element_data[20:24])[0]

            xdata_values = []
            start = 24
            for i in range(num_xdata_points):
                end = start + 8
                xdata_values.append(struct.unpack("<d", element_data[start:end])[0])
                start = end
            xdat_data.append([num_point, y_coord, x_coord, num_xdata_points, xdata_values])
        elif magic_bytes == 'MLOV':
            pass
        else:
            continue

    return [volm_ftoc, volm_ttoc, volm_vdef, x_channel_names, data_channels_names, data_channels_units, voff_data,
            vset_data, vnam_data, vdat_data, xdat_data]


def ARDF_get_IMAG(fd, imag_entry):
    imag_entry_offset = imag_entry[1]
    fd.seek(imag_entry_offset)
    imag_entry = ARDF_get_next_element(fd)
    imag_ftoc = ARDF_parse_FTOC(fd, imag_entry[0], imag_entry[1])
    imag_ttoc_entry = ARDF_get_next_element(fd)
    imag_ttoc = ARDF_parse_TTOC(fd, imag_ttoc_entry[0], imag_ttoc_entry[1])

    idef_entry = ARDF_get_next_element(fd)[1]

    magic_bytes = struct.unpack("<4s", idef_entry[0:4])[0].decode("ascii")
    unk1 = struct.unpack("<L", idef_entry[4:8])[0]
    x_pixcount = struct.unpack("<L", idef_entry[8:12])[0]
    y_pixcount = struct.unpack("<L", idef_entry[12:16])[0]
    unk2 = struct.unpack("<d", idef_entry[16:24])[0]
    unk3 = struct.unpack("<d", idef_entry[24:32])[0]
    x_step = struct.unpack("<d", idef_entry[32:40])[0]
    y_step = struct.unpack("<d", idef_entry[40:48])[0]
    x_unit = struct.unpack("<32s", idef_entry[48:80])[0].decode("ascii")
    y_unit = struct.unpack("<32s", idef_entry[80:112])[0].decode("ascii")
    chan_name = struct.unpack("<32s", idef_entry[112:144])[0].decode("ascii")
    z_unit = struct.unpack("<32s", idef_entry[144:176])[0].decode("ascii")

    imag_idef = [chan_name, x_pixcount, y_pixcount, x_step, y_step, x_unit, y_unit, z_unit]

    ibox_entry = ARDF_get_next_element(fd)[1]
    magic_bytes = struct.unpack("<4s", ibox_entry[0:4])[0].decode("ascii")
    unk1 = struct.unpack("<L", ibox_entry[4:8])[0]
    total_size = struct.unpack("<L", ibox_entry[8:12])[0]
    unk2 = struct.unpack("<L", ibox_entry[12:16])[0]
    num_lines = struct.unpack("<L", ibox_entry[16:20])[0]
    line_size = struct.unpack("<L", ibox_entry[20:24])[0]

    imag_ibox = [num_lines]

    imag_data = []

    for i in range(num_lines):
        idat_entry = ARDF_get_next_element(fd)[1]
        magic_bytes = struct.unpack("<4s", idat_entry[0:4])[0].decode("ascii")
        unk1 = struct.unpack("<L", idat_entry[4:8])[0]
        line_data = struct.unpack("<" + str(x_pixcount) + "f", idat_entry[8:8 + 4 * x_pixcount])
        imag_data.append(line_data)

    return [imag_ftoc, imag_ttoc, imag_idef, imag_ibox, imag_data]


def ARDF_parse_THMB(fd, thmb_entry):
    thmb_entry_offset = thmb_entry[1]
    fd.seek(thmb_entry_offset)
    thmb_entry = ARDF_get_next_element(fd)[1]

    magic_bytes = struct.unpack("<4s", thmb_entry[0:4])[0].decode("ascii")
    unk1 = struct.unpack("<L", thmb_entry[4:8])[0]
    x_size = struct.unpack("<L", thmb_entry[8:12])[0]
    y_size = struct.unpack("<L", thmb_entry[12:16])[0]
    bit_depth = struct.unpack("<L", thmb_entry[16:20])[0]
    unk2 = struct.unpack("<L", thmb_entry[20:24])[0]
    nbytes = int(x_size * y_size * (bit_depth / 8))
    bin_data = struct.unpack("<" + str(nbytes) + "s", thmb_entry[24:24 + nbytes])[0]

    return [x_size, y_size, bin_data]


def ardf2hdf5(filename, filepath=None):
    gc.enable()
    ftoc, ttoc, fd = ARDF_load_file(filename)

    # print("======================================")
    # print("Going through global FTOC:")
    for f in ftoc:
        # print(f)
        if f[0] == 'IMAG':
            # print("Found IMAG header, parsing:")
            imag_ftoc, imag_ttoc, imag_idef, imag_ibox, imag_data = ARDF_get_IMAG(fd, f)
            for imag_f in imag_ftoc:
                if imag_f[0] == 'NEXT':
                    # print("--> Ignoring NEXT header for now.")
                    pass
                elif imag_f[0] == 'THMB':
                    # print("--> Parsing THMB (thumbnail):")
                    [x, y, d] = ARDF_parse_THMB(fd, imag_f)
                    d_img = np.array(bytearray(d), dtype=np.byte).reshape((x, y))
                else:
                    print("Found '" + str(imag_f[0]) + "' header. What is it?")
            for imag_t in imag_ttoc:
                if imag_t[0] == 'TOFF':
                    # print("--> Found TOFF, parsing TEXT.")
                    pass
                else:
                    print("Found '" + str(imag_t[0]) + "' header. What is it?")
        elif f[0] == 'VOLM':
            # print("Found VOLM header, parsing:")
            volm_ftoc, volm_ttoc, volm_vdef, x_channel_names, data_channels_names, data_channels_units, voff_data, vset_data, vnam_data, vdat_data, xdat_data = ARDF_get_VOLM(fd, f)

            # with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
            #     metadatagrp = f.create_group("metadata")
            #     f.create_group("process")
            #
            #     if filepath is not None:
            #         datagrp = f.create_group("datasets/" + filepath.split('.')[0])
            #         datagrp.attrs.__setattr__('type', filepath.split('.')[-1])
            #     else:
            #         datagrp = f.create_group("datasets/" + filename.split('.')[0])
            #         datagrp.attrs.__setattr__('type', filename.split('.')[-1])
            #
            #     label_list = list(map(lambda x: x.decode('utf-8').replace('\x00', ''), data_channels_names))
            #     y_size = volm_vdef[2]
            #     x_size = volm_vdef[3]
            #     z_size = vdat_data[0][6] - vdat_data[0][5]
            #     numchans = len(label_list)
            #     data_matrix = np.ndarray(shape=(numchans, y_size, x_size, z_size))
            #     for vd in vdat_data:
            #         try:
            #             ypos = vd[1]
            #             xpos = vd[2]
            #             chanindex = vd[4]
            #             data = vd[9][vd[5]:vd[6]]
            #             data_matrix[chanindex, ypos, xpos, :] = data[:]
            #         except:
            #             pass
            #
            #     for i, k in enumerate(label_list):
            #
            #         datagrp.create_dataset(k, data=data_matrix[i, :, :, :])
            #         try:
            #             datagrp[k].attrs['unit'] = data_channels_units[i].decode('utf-8').replace('\x00', '')
            #         except:
            #             if b'\xb0' in data_channels_units[i]:
            #                 datagrp[k].attrs['unit'] = 'deg'
            #             else:
            #                 datagrp[k].attrs['unit'] = 'unknown'
            #         datagrp[label_list[i]].attrs['name'] = k
            #         datagrp[label_list[i]].attrs['shape'] = data_matrix.shape
            #         # datagrp[label_list[i]].attrs['size'] = (fastsize,slowsize)
            #         # datagrp[label_list[i]].attrs['offset'] = (xoffset,yoffset)

        else:
            print("Found '" + str(f[0]) + "' header. What is it?")
    # print("======================================")
    # print("Going through global TTOC:")
    for t in ttoc:
        if t[0] == 'TOFF':
            meta = ARDF_get_TEXT_from_TOFF(fd, t).decode('windows-1252')
            metalist = meta.split('\r')
            metalist = [i.encode('utf8') for i in metalist]
        else:
            print("Found '" + str(t[0]) + "' header. What is it?")

        # with h5py.File(filename.split('.')[0] + ".hdf5", "a") as f:
        #     if filepath is not None:
        #         f["metadata"].create_dataset(filepath.split('.')[0], data=metalist)
        #     else:
        #         f["metadata"].create_dataset(filename.split('.')[0], data=metalist)

    gc.disable()
    var = len(data_channels_names)
    raw = []
    defl = []
    zsnsr = []
    
    for i in range(int(len(vdat_data)/var)):
        raw.append(np.array(vdat_data[i*var + 0][9]))
        defl.append(np.array(vdat_data[i*var + 1][9]))
        zsnsr.append(np.array(vdat_data[i*var + 2][9]))
        
    metadict = {}     
    for item in metalist: # IG added
        newitem = str(item)[2:-1]
        split = newitem.split(':')
        name = split[0]
        value = ':'.join(split[1:])

        metadict[name] = value
        
    return [[raw, defl, zsnsr], metadict] # PN added, IG modified