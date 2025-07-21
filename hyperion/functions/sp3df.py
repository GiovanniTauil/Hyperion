import dataclasses
import datetime
import enum
import math
import re
import typing
import os
from decimal import Decimal, InvalidOperation
import pandas as pd

class TimeSystem(enum.Enum):
    GPS = b"GPS"
    GLO = b"GLO"
    GAL = b"GAL"
    QZS = b"QZS"
    BDT = b"BDT"
    IRN = b"IRN"
    UTC = b"UTC"
    TAI = b"TAI"

class Version(enum.Enum):
    C = b"c"
    D = b"d"

class FileType(enum.Enum):
    GPS = b"G"
    MIXED = b"M"
    GLONASS = b"R"
    LEO = b"L"
    SBAS = b"S"
    IRNSS = b"I"
    GALILEO = b"E"
    BEIDOU = b"B"
    QZSS = b"J"

@dataclasses.dataclass
class Record:
    time: datetime.datetime
    position: tuple[Decimal, Decimal, Decimal]  # m
    position_std: typing.Optional[tuple[float, float, float]] = None  # m
    velocity: typing.Optional[tuple[Decimal, Decimal, Decimal]] = None  # m/s
    velocity_std: typing.Optional[tuple[float, float, float]] = None  # m/s
    clock: typing.Optional[Decimal] = None  # s
    clock_std: typing.Optional[float] = None  # s
    clock_rate: typing.Optional[Decimal] = None  # s/s
    clock_rate_std: typing.Optional[float] = None  # s/s
    clock_event: bool = False
    clock_predicted: bool = False
    xy_correlation: typing.Optional[float] = None
    xz_correlation: typing.Optional[float] = None
    xc_correlation: typing.Optional[float] = None
    yz_correlation: typing.Optional[float] = None
    yc_correlation: typing.Optional[float] = None
    zc_correlation: typing.Optional[float] = None
    xy_velocity_correlation: typing.Optional[float] = None
    xz_velocity_correlation: typing.Optional[float] = None
    xc_velocity_correlation: typing.Optional[float] = None
    yz_velocity_correlation: typing.Optional[float] = None
    yc_velocity_correlation: typing.Optional[float] = None
    zc_velocity_correlation: typing.Optional[float] = None

@dataclasses.dataclass
class Satellite:
    id: bytes
    accuracy: typing.Optional[float] = None  # m
    records: list[Record] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Product:
    version: Version
    file_type: FileType
    time_system: TimeSystem
    data_used: bytes
    coordinate_system: bytes
    orbit_type: bytes
    agency: bytes
    comments: list[bytes]
    satellites: list[Satellite]

    @classmethod
    def from_bytes(cls, data: bytes):
        id_to_patterns = {
            b"0": (
                re.compile(
                    rb"^#([cd])([PV])(\d{4}) ( \d|\d{2}) ( \d|\d{2}) ( \d|\d{2}) ( \d|\d{2}) ((?: \d|\d{2})\.\d{8}) ([ \d]{7}) (.{5}) (.{5}) (.{3}) (.{4})(?:$|\s+$)"
                ),
            ),
            b"1": (
                re.compile(
                    rb"^## ([ \d]{4}) ([ \.\d]{6}\.\d{8}) ([ \d]{5}\.\d{8}) ([ \d]{5}) ([ \d]{1}\.\d{13})(?:$|\s+$)"
                ),
            ),
            b"+0": (re.compile(rb"^\+  ([ \d]{3})   ((?:\w\d{2})*).*(?:$|\s+$)"),),
            b"+": (re.compile(rb"^\+        ((?:\w\d{2})*).*(?:$|\s+$)"),),
            b"++": (
                re.compile(rb"^\+\+       ((?:[ \d]{3})*)(?:$|\s+$)"),
                re.compile(rb"^\+\+$"),
            ),
            b"c0": (
                re.compile(
                    rb"^%c ([\w ]{2}) cc ([\w ]{3}) ccc cccc cccc cccc.cccc ccccc ccccc ccccc ccccc(?:$|\s+$)"
                ),
            ),
            b"c1": (
                re.compile(
                    rb"^%c cc cc ccc ccc cccc cccc cccc.cccc ccccc ccccc ccccc ccccc(?:$|\s+$)"
                ),
            ),
            b"f0": (
                re.compile(
                    rb"^%f ([ \d]{2}\.\d{7}) ([ \d]{2}\.\d{9})  0\.00000000000  0\.000000000000000(?:$|\s+$)"
                ),
            ),
            b"f1": (
                re.compile(
                    rb"^%f  0\.0000000  0\.000000000  0\.00000000000  0\.000000000000000(?:$|\s+$)"
                ),
            ),
            b"i": (
                re.compile(
                    rb"^%i    0    0    0    0      0      0      0      0         0(?:$|\s+$)"
                ),
            ),
            b"/": (re.compile(rb"^/\*($| .*)(?:$|\s+$)"),),
            b"*": (
                re.compile(
                    rb"^\*  (\d{4}) ( \d|\d{2}) ( \d|\d{2}) ( \d|\d{2}) ( \d|\d{2}) ((?: \d|\d{2})\.\d{8})(?:$|\s+$)"
                ),
            ),
            b"p": (
                re.compile(
                    rb"^P(\w\d{2})([ \d-]{7}\.\d{6})([ \d-]{7}\.\d{6})([ \d-]{7}\.\d{6})(?:([ \d-]{7}\.\d{6}| {14})(?: ([ \d]{2}) ([ \d]{2}) ([ \d]{2}) ([ \d]{3}) ([ \w])([ \w])|)|)(?:$|\s+$)"
                ),
            ),
            b"ep": (
                re.compile(
                    rb"^EP  ([ -0-9]{4}) ([ -0-9]{4}) ([ -0-9]{4}) ([ -0-9]{7}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8})(?:$|\s+$)$"
                ),
            ),
            b"v": (
                re.compile(
                    rb"^V(\w\d{2})\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+|999999\.999999)?\s*$"
                ),
            ),
            b"ev": (
                re.compile(
                    rb"^EV  ([ -0-9]{4}) ([ -0-9]{4}) ([ -0-9]{4}) ([ -0-9]{7}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8}) ([ -0-9]{8})(?:$|\s+$)$"
                ),
            ),
            b"eof": (re.compile(rb"^EOF\s*$"),),
        }

        state = 0
        product: typing.Optional[Product] = None
        start = None
        satellites_count = 0
        satellite_index = 0
        includes_velocities = False
        epochs = 0
        epoch_interval = datetime.timedelta()
        epoch_index = 0
        epoch: typing.Optional[datetime.datetime] = None
        position_base = Decimal('2.0')
        clock_base = Decimal('2.0')

        lines = data.split(b"\n")
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            if not line:
                continue
            id: typing.Optional[bytes] = None
            if state < 9:
                if idx == 1:
                    id = b"0"
                elif idx == 2:
                    id = b"1"
                elif idx == 3:
                    id = b"+0"
                    state = 1
                else:
                    if state == 1:
                        if line.startswith(b"+ "):
                            id = b"+"
                        else:
                            state = 2
                            id = b"++"
                    elif state == 2:
                        if line.startswith(b"++"):
                            id = b"++"
                        else:
                            state = 3
                            id = b"c0"
                    elif state == 3:
                        state = 4
                        id = b"c1"
                    elif state == 4:
                        state = 5
                        id = b"f0"
                    elif state == 5:
                        state = 6
                        id = b"f1"
                    elif state in (6, 7):
                        state += 1
                        id = b"i"
                    elif state == 8:
                        if line.startswith(b"/* "):
                            id = b"/"
                        elif line.startswith(b"* "):
                            state = 9
                            id = b"*"
                        elif len(line.strip()) == 0:
                            continue
                        else:
                            raise Exception(f"unexpected line in header {line.decode()}")
            else:
                if line.startswith(b"* "):
                    id = b"*"
                elif line.startswith(b"P"):
                    id = b"p"
                elif line.startswith(b"V"):
                    id = b"v"
                elif line.startswith(b"EP"):
                    id = b"ep"
                elif line.startswith(b"EV"):
                    id = b"ev"
                elif line.startswith(b"EOF"):
                    id = b"eof"
                elif line.startswith(b"/*"):
                    id = b"/"
                elif len(line.strip()) == 0:
                    continue
                else:
                    continue

            match: typing.Optional[re.Match[bytes]] = None
            if id in id_to_patterns:
                for pattern in id_to_patterns[id]:
                    match = pattern.match(line)
                    if match is not None:
                        break
                if match is None and id not in (b"++",):
                    raise Exception(
                        f'SP3 line {idx} ("{line}") did not match the pattern "{id}"'
                    )
            if id == b"0":
                product = Product(
                    version=Version(match[1]),
                    file_type=None,  # type: ignore
                    time_system=None,  # type: ignore
                    data_used=match[10].strip(),
                    coordinate_system=match[11].strip(),
                    orbit_type=match[12].strip(),
                    agency=match[13].strip(),
                    comments=[],
                    satellites=[],
                )
                includes_velocities = match[2] == b"V"
                epochs = int(match[9])
                start = datetime.datetime(
                    year=int(match[3]),
                    month=int(match[4]),
                    day=int(match[5]),
                    hour=int(match[6]),
                    minute=int(match[7]),
                    second=math.floor(float(match[8])),
                    microsecond=round(
                        (float(match[8]) - math.floor(float(match[8]))) * 1e6
                    ),
                    tzinfo=datetime.timezone.utc,
                )
            elif id == b"1":
                assert product is not None
                epoch_interval = datetime.timedelta(seconds=float(match[3]))
            elif id == b"+0" or id == b"+":
                if id == b"+0":
                    satellites_count = int(match[1])
                assert product is not None
                packed_ids = match[1 if id == b"+" else 2]
                for slice_start in range(0, len(packed_ids), 3):
                    sat_id = packed_ids[slice_start : slice_start + 3]
                    if sat_id == b'  0':
                        continue
                    product.satellites.append(
                        Satellite(
                            id=sat_id,
                            accuracy=None,
                            records=[],
                        )
                    )
            elif id == b"++":
                assert product is not None
                if match is None or len(match.groups()) < 1 or match[1] is None:
                    pass
                else:
                    for slice_start in range(0, len(match[1]), 3):
                        stripped_slice = match[1][slice_start : slice_start + 3].strip()
                        if len(stripped_slice) > 0:
                            exponent = int(stripped_slice)
                            if satellite_index < len(product.satellites):
                                product.satellites[satellite_index].accuracy = (
                                    None
                                    if exponent == 0
                                    else ((2.0**exponent) / 1000.0)
                                )
                                satellite_index += 1
                            elif exponent > 0:
                                raise Exception(
                                    "there are more accuracy fields than satellites"
                                )
            elif id == b"c0":
                assert product is not None
                product.file_type = FileType(match[1].strip())
                product.time_system = TimeSystem(match[2])
            elif id == b"f0":
                position_base = Decimal(match[1].decode())
                clock_base = Decimal(match[2].decode())
            elif id == b"/":
                assert product is not None
                comment = match[1].strip()
                if len(comment) > 0:
                    product.comments.append(comment)
            elif id == b"*":
                assert product is not None
                if epoch_index > 0:
                    assert satellite_index == len(product.satellites)
                year = int(match[1])
                month = int(match[2])
                day = int(match[3])
                hour = int(match[4])
                minute = int(match[5])
                sec = float(match[6])
                epoch = datetime.datetime(
                    year=year,
                    month=month,
                    day=day,
                    hour=hour,
                    minute=minute,
                    second=math.floor(sec),
                    microsecond=round((sec - math.floor(sec)) * 1e6),
                    tzinfo=datetime.timezone.utc,
                )
                epoch_index += 1
                satellite_index = 0
                continue
            elif id == b"p":
                assert product is not None and epoch is not None
                prn = match[1]
                assert satellite_index < len(product.satellites)
                assert prn == product.satellites[satellite_index].id
                x_str = match[2].decode().strip()
                y_str = match[3].decode().strip()
                z_str = match[4].decode().strip()
                clock_str = match[5].decode().strip() if match[5] else ''
                try:
                    dec_x = Decimal(x_str) if x_str else Decimal('0')
                    dec_y = Decimal(y_str) if y_str else Decimal('0')
                    dec_z = Decimal(z_str) if z_str else Decimal('0')
                except InvalidOperation:
                    dec_x = dec_y = dec_z = Decimal('0')
                try:
                    dec_clock = Decimal(clock_str) if clock_str else None
                except InvalidOperation:
                    dec_clock = None
                if dec_clock == Decimal('999999.999999'):
                    dec_clock = None
                position_std = None
                clock_std = None
                clock_event = False
                clock_predicted = False
                if match[6] and match[7] and match[8] and match[6].strip() and match[7].strip() and match[8].strip():
                    position_std = (
                        float(position_base ** Decimal(match[6].decode().strip())) * 0.001,
                        float(position_base ** Decimal(match[7].decode().strip())) * 0.001,
                        float(position_base ** Decimal(match[8].decode().strip())) * 0.001,
                    )
                if match[9] and match[9].strip():
                    clock_std = float(clock_base ** Decimal(match[9].decode().strip())) * 1e-12
                if match[10] == b'E':
                    clock_event = True
                if match[11] == b'P':
                    clock_predicted = True
                rec = Record(
                    time=epoch,
                    position=(
                        dec_x * Decimal('1000'),
                        dec_y * Decimal('1000'),
                        dec_z * Decimal('1000'),
                    ),
                    position_std=position_std,
                    velocity=None,
                    velocity_std=None,
                    clock=dec_clock * Decimal('1e-6') if dec_clock is not None else None,
                    clock_std=clock_std,
                    clock_rate=None,
                    clock_rate_std=None,
                    clock_event=clock_event,
                    clock_predicted=clock_predicted,
                )
                product.satellites[satellite_index].records.append(rec)
                satellite_index += 1
            elif id == b"v":
                assert product is not None
                prn = match[1]
                sat_idx = None
                for idx2, s in enumerate(product.satellites):
                    if s.id == prn:
                        sat_idx = idx2
                        break
                if sat_idx is None:
                    continue
                rec = product.satellites[sat_idx].records[-1]
                vx_str = match[2].decode().strip()
                vy_str = match[3].decode().strip()
                vz_str = match[4].decode().strip()
                rate_str = match[5].decode().strip() if match.lastindex and match.lastindex >= 5 and match[5] else ""
                try:
                    # Vx, Vy, Vz in decimeters/second -> meters/second
                    dec_vx = Decimal(vx_str) / Decimal('10') if vx_str else Decimal('0')
                    dec_vy = Decimal(vy_str) / Decimal('10') if vy_str else Decimal('0')
                    dec_vz = Decimal(vz_str) / Decimal('10') if vz_str else Decimal('0')
                except InvalidOperation:
                    dec_vx = dec_vy = dec_vz = Decimal('0')
                rec.velocity = (
                    dec_vx,
                    dec_vy,
                    dec_vz,
                )
                # dT: 10^-4 microseconds/second, to seconds/second
                if rate_str and rate_str != "999999.999999":
                    try:
                        # First: microseconds/second
                        dec_rate = Decimal(rate_str) * Decimal("1e-4")
                        # microseconds to seconds
                        rec.clock_rate = dec_rate * Decimal("1e-6")
                    except InvalidOperation:
                        rec.clock_rate = None
                else:
                    rec.clock_rate = None
            elif id == b"ep":
                assert product is not None
                sat_idx = satellite_index - 1
                if sat_idx < 0 or sat_idx >= len(product.satellites):
                    continue
                rec = product.satellites[sat_idx].records[-1]
                sdev_x_str = line[4:8].decode().strip()
                sdev_y_str = line[9:13].decode().strip()
                sdev_z_str = line[14:18].decode().strip()
                sdev_c_str = line[19:26].decode().strip()
                corr_xy_str = line[27:35].decode().strip()
                corr_xz_str = line[36:44].decode().strip()
                corr_xc_str = line[45:53].decode().strip()
                corr_yz_str = line[54:62].decode().strip()
                corr_yc_str = line[63:71].decode().strip()
                corr_zc_str = line[72:80].decode().strip()
                sdev_x = int(sdev_x_str) if sdev_x_str else 0
                sdev_y = int(sdev_y_str) if sdev_y_str else 0
                sdev_z = int(sdev_z_str) if sdev_z_str else 0
                rec.position_std = (
                    sdev_x * 0.001,
                    sdev_y * 0.001,
                    sdev_z * 0.001,
                )
                rec.clock_std = int(sdev_c_str) * 1e-12 if sdev_c_str else None
                rec.xy_correlation = int(corr_xy_str) / 10000000.0 if corr_xy_str else None
                rec.xz_correlation = int(corr_xz_str) / 10000000.0 if corr_xz_str else None
                rec.xc_correlation = int(corr_xc_str) / 10000000.0 if corr_xc_str else None
                rec.yz_correlation = int(corr_yz_str) / 10000000.0 if corr_yz_str else None
                rec.yc_correlation = int(corr_yc_str) / 10000000.0 if corr_yc_str else None
                rec.zc_correlation = int(corr_zc_str) / 10000000.0 if corr_zc_str else None
            elif id == b"ev":
                assert product is not None
                sat_idx = satellite_index - 1
                if sat_idx < 0 or sat_idx >= len(product.satellites):
                    continue
                rec = product.satellites[sat_idx].records[-1]
                sdev_vx_str = line[4:8].decode().strip()
                sdev_vy_str = line[9:13].decode().strip()
                sdev_vz_str = line[14:18].decode().strip()
                sdev_r_str = line[19:26].decode().strip()
                corr_xy_str = line[27:35].decode().strip()
                corr_xz_str = line[36:44].decode().strip()
                corr_xr_str = line[45:53].decode().strip()
                corr_yz_str = line[54:62].decode().strip()
                corr_yr_str = line[63:71].decode().strip()
                corr_zr_str = line[72:80].decode().strip()
                sdev_vx = int(sdev_vx_str) if sdev_vx_str else 0
                sdev_vy = int(sdev_vy_str) if sdev_vy_str else 0
                sdev_vz = int(sdev_vz_str) if sdev_vz_str else 0
                rec.velocity_std = (
                    sdev_vx * 1e-7,
                    sdev_vy * 1e-7,
                    sdev_vz * 1e-7,
                )
                rec.clock_rate_std = int(sdev_r_str) * 1e-16 if sdev_r_str else None
                rec.xy_velocity_correlation = int(corr_xy_str) / 10000000.0 if corr_xy_str else None
                rec.xz_velocity_correlation = int(corr_xz_str) / 10000000.0 if corr_xz_str else None
                rec.xc_velocity_correlation = int(corr_xr_str) / 10000000.0 if corr_xr_str else None
                rec.yz_velocity_correlation = int(corr_yz_str) / 10000000.0 if corr_yz_str else None
                rec.yc_velocity_correlation = int(corr_yr_str) / 10000000.0 if corr_yr_str else None
                rec.zc_velocity_correlation = int(corr_zr_str) / 10000000.0 if corr_zr_str else None
            elif id == b"eof":
                break

        assert product is not None
        assert epoch_index == epochs
        if epoch_index > 0:
            assert satellite_index == len(product.satellites)
        return product

    @classmethod
    def from_file(cls, path: typing.Union[str, bytes, os.PathLike]):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as file:
            return cls.from_bytes(file.read())

def read_sp3_to_dataframe(
    file_path: typing.Union[str, bytes, os.PathLike], 
    position_decimals: int = 3, 
    clock_decimals: int = 12,
) -> pd.DataFrame:
    product = Product.from_file(file_path)
    data = []
    quantize_pos = Decimal(f'1.{"0" * position_decimals}')
    quantize_clk = Decimal(f'1.{"0" * clock_decimals}')
    have_velocity = any(
        any(rec.velocity is not None for rec in sat.records) for sat in product.satellites
    )
    have_clockrate = any(
        any(rec.clock_rate is not None for rec in sat.records) for sat in product.satellites
    )
    for sat in product.satellites:
        prn = sat.id.decode('utf-8').strip()
        for rec in sat.records:
            if all(p == Decimal('0') for p in rec.position) or rec.clock is None:
                continue
            try:
                pos_x = rec.position[0].quantize(quantize_pos)
                pos_y = rec.position[1].quantize(quantize_pos)
                pos_z = rec.position[2].quantize(quantize_pos)
                clock = rec.clock.quantize(quantize_clk)
            except (InvalidOperation, AttributeError):
                continue
            row = {
                'Epoch': rec.time,
                'PRN': prn,
                'X(m)': float(pos_x),
                'Y(m)': float(pos_y),
                'Z(m)': float(pos_z),
                'Clock(s)': float(clock),
            }
            if have_velocity:
                vx = vy = vz = None
                if rec.velocity is not None:
                    vx = rec.velocity[0]
                    vy = rec.velocity[1]
                    vz = rec.velocity[2]
                row['Vx(m/s)'] = str(vx) if vx is not None else None
                row['Vy(m/s)'] = str(vy) if vy is not None else None
                row['Vz(m/s)'] = str(vz) if vz is not None else None
            if have_clockrate:
                row['ClockRate(s/s)'] = str(rec.clock_rate) if rec.clock_rate is not None else None
            data.append(row)
    df = pd.DataFrame(data)
    if not df.empty:
        df['Clock(s)'] = df['Clock(s)'].map(lambda x: f"{x:.{clock_decimals}f}")
    return df
