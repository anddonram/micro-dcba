import csv
import matplotlib.pyplot as plt


def get_data(filename):
    sim='SIMULATION'
    step=    ' STEP'
    env=        ' ENVIRONMENT'
    mem=        ' MEMBRANE'
    obj=        ' OBJECT'
    mult=        ' MULTIPLICITY'

    vals={}

    num_sims=0
    max_obj=0

    especies={"1":"Quebrantahuesos",
    "2":"Rebeco pirenaico",
    "3":"Ciervo rojo hembra",
    "4":"Ciervo rojo macho",
    "5":"Gamo",
    "6":"Corzo",
    "7":"Oveja",
    "B":"B",
    "C":"C"
    }

    with open(filename) as file:
        reader=csv.DictReader(file,delimiter=",")
        print(reader.fieldnames)
        for line in reader:
            esp=line[obj]
            if esp.find(",")!=-1:
                esp=esp[0:esp.find(",")]


            tupl=(line[env],line[mem],esp)
            step_dict=vals.setdefault(tupl,{})

            step_val=int(line[step])
            step_dict.setdefault(step_val,0)
            step_dict[step_val]+=int(line[mult])
            num_sims=max(num_sims,int(line[sim]))

    num_sims+=1
    #print(vals)
    print(num_sims)

    for tupl in vals:

        vals_dict=vals[tupl]
        vals_key=sorted(vals_dict)
        plt.clf()
        plt.plot([x/3 for x in vals_key],[vals_dict[x]/num_sims for x in vals_key], '-o')
        plt.ylabel("Población "+especies[tupl[2][-1]])
        plt.xlabel("Tiempo (años)")

        plt.savefig('especie_evol_'+tupl[2]+'.png')




if __name__=="__main__":
    get_data("bv_model_bwmc12.bin_output.csv")
