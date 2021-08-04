
class Parser():
	def __init__(self, filename):
		self.filename = filename
		self.updated = None


	@property
	def input(self):
		with open(self.filename, 'r') as f:
			rad = f.readlines()
		return rad

	@property
	def header_index(self):
		for i, e in enumerate(self.input):
			if 'Atoms' in e:
				return i

	@property
	def epilogue_index(self):
		for i, e in enumerate(self.input):
			if 'Velocities' in e:
				return i

	@property
	def header(self):
		return self.input[:self.header_index + 2]
	
	@property
	def coordinates(self):
		bulk =  self.input[self.header_index+2: self.epilogue_index-1]
		bulk = [i.split() for i in bulk]
		return [[float(i[4]), float(i[5]), float(i[6])] for i in bulk]

	@property
	def num_atoms(self):
		return len(self.coordinates)

	@property
	def epilogue(self):
		return self.input[self.epilogue_index-1:]

	
	def update_coordinate(self, new):
		new = [[new[f'x_{i}_{j}'] for j in range(3)] for i in range(self.num_atoms)]
		
		updated = []
		for i in range(self.num_atoms):
			updated.append(f"{i+1} 0 1 0 {new[i][0]} {new[i][1]} {new[i][2]} 0 0 0\n")

		
		self.updated = updated

	def list_to_dic(self):
		return {f'x_{i}_{j}':l[j] for i, l in enumerate(self.coordinates) for j in range(3)}

	def write(self, outfile):
		text = self.header + self.updated + self.epilogue
		with open(outfile, 'w') as f:
			for l in text:
				f.write(l)


