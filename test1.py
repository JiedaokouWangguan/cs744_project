from struct import pack, unpack

b = pack('f', 342342342300)
print(342342342300)
c = unpack('I', b)[0]

# print(unpack('i', b)[0])
print(c)
x1 = 0x000000FF & c
x2 = (0X0000FF00 & c) >> 8
x3 = (0x00FF0000 & c) >> 16
x4 = (0xFF000000 & c) >> 24




r = 0
r = x1 | r
r = (x2 << 8) | r
r = (x3 << 16) | r
r = (x4 << 24) | r
print(r)

r1 = pack('I', r)
r2 = unpack('f', r1)[0]
print(r2)





