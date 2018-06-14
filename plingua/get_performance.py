import csv

def get_data(filename):

    start="* Time in parts"
    speedup="Speedup on phases: "
    profgpu="Profiling GPU: "
    profcpu="Profiling CPU: "
    total="Total time: "

    data=[["Aceleración fase 1","Aceleración fase 2","Aceleración fase 3","Aceleración fase 4",
    "% GPU fase 1","% GPU fase 2","% GPU fase 3","% GPU fase 4",
    "% CPU fase 1","% CPU fase 2","% CPU fase 3","% CPU fase 4",
    "Tiempo GPU", "Tiempo CPU", "Aceleración"]]

    sim_data=None
    with open(filename) as file:

        for line in file:
            if line.startswith(start):
                workline=line[len(start):].strip()
                sim_data=[]
                sim_data.extend([w.split("=")[1].replace(".",",") for w in workline.split(",") if w])
            elif line.startswith(speedup):
                workline=line[len(speedup):].strip()
                sim_data.extend([w.split("=")[1].replace(".",",") for w in workline.split(",") if w])
            elif line.startswith(profgpu):
                workline=line[len(profgpu):].strip()
                sim_data.extend([w.split("=")[1].replace(".",",") for w in workline.split(",") if w])
            elif line.startswith(profcpu):
                workline=line[len(profcpu):].strip()
                sim_data.extend([w.split("=")[1].replace(".",",") for w in workline.split(",") if w])
            elif line.startswith(total):
                workline=line[len(total):].strip()
                sim_data.extend([w.split("=")[1].replace(".",",") for w in workline.split(",") if w])
                data.append(sim_data)
    with open(filename+".csv", "w") as file:
        writer=csv.writer(file)
        for d in data:
            writer.writerow(d)
if __name__=="__main__":
    get_data("res_950.txt")
    get_data("res_tesla.txt")
