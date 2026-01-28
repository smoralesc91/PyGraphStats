# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 02:40:55 2021

@author: ernestito
"""

import csv # Valores Separados por Coma
import re # Expresiones Regulares
import numpy as np
import math 


def leerCoordenadasSA(archivo):
    
    coor = [] # Arreglo bidimensional de coordenadas: Ncoor X 3
    with open(archivo) as fid:
        reader = csv.reader(fid)
        ln = 1 # Numero de linea
        for row in reader:
            if ln<=9: # Imprime solamente las primeras 9 lineas del archivo
                print( ln, ' : ', row )
            if ln == 4: # Ejemplo de linea 4: '# rows: 760'
                x = re.findall("[0-9]", row[0]) # Busca numeros: '7','6','0'
                xstr = ''.join([str(elem) for elem in x]) # Concatena '7','6','0' -> '760'
                NPT = int(float(xstr)) # Convierte de cadena a entero '760' -> 760
            if ln>=6: # Ejemplo de linea 6: 'b9.6803913764472238e-05b4.9315550283240597e-05b1.9840037113944567e-05'
                if len( row ) > 0: # Si la linea tiene longitud mayor que cero
                    x = re.split("\s", row[0]) # Separa segun espacio en blanco: 'b#1b#2b#3' - > ['','#1','#2','#3']
                    coor.append( [float(x[1]),float(x[2]),float(x[3])] ) # Convierte de cadena a flotante y pega al arreglo de coordenadas
            ln += 1 # Siguiente linea
    print(' ')
            
    return np.array( coor, dtype=np.float64 )


def leerConexionesSA(archivo):
    
    conn = []
    with open(archivo) as fid:
        reader = csv.reader(fid)
        ln = 1
        for row in reader:
            if ln<=9:
                print( ln, ' : ', row )
            if ln>=1:              
                if len( row )>0:
                    conn.append( [int(float(row[0]))-1,int(float(row[1]))-1] )
            ln += 1
    print(' ')
            
    return conn


def longitudesGargantas( conexiones, coordenadas ):
    
    NTC = len( conexiones )
    dist = np.zeros( (NTC), dtype=np.float64 )
    for i in range(NTC):
        P1 = coordenadas[ conexiones[i][0] ]
        P2 = coordenadas[ conexiones[i][1] ]
        dist[i] = np.linalg.norm(P2 - P1)
        
    return dist
        

def coordenadasMaxMin( coordenadas ):

    NPT = len( coordenadas )

    coorMin = coordenadas[0].copy()
    coorMax = coordenadas[0].copy()

    for i in range(NPT):
    
        coor = coordenadas[i]
        for j in range(3):
            
            if coor[j]<coorMin[j]:
                coorMin[j] = coor[j]
                
            if coor[j]>coorMax[j]:
                coorMax[j] = coor[j]
                
    return [ np.array( coorMin, dtype=np.float64), np.array( coorMax, dtype=np.float64 ) ]


def frontera( eje, coordenadas, r ):
    
    nodosIzq = []
    nodosDer = []
    
    coorMin, coorMax = coordenadasMaxMin( coordenadas )
    delta = ( coorMax[eje] - coorMin[eje] ) * r
    
    NPT = len( coordenadas )

    for i in range(NPT):
        
        coor = coordenadas[i]
        
        if coor[eje] < ( coorMin[eje]+delta ):
            nodosIzq.append( i )
            
        if coor[eje] > ( coorMax[eje]-delta ):
            nodosDer.append( i )
        
    return [ nodosIzq, nodosDer ]


def conexionesRuta( ruta, conexiones ):
    
    for i in range( len(ruta)-1 ):
        conexiones.append( [ ruta[i],ruta[i+1] ] ) 
    return 0


def leerUrna( archivo ):
    
    urna = []
    with open( archivo ) as fid:
        spamreader = csv.reader(fid)
        ln = 1
        for row in spamreader:
            if ln <= 9:
                print( ln, ' : ', row )
            if ln >= 2:
                urna.append( float(row[1]) )
            ln = ln+1
    
    return np.array( urna, dtype=np.float64 )


def leerUrnaMatlab( archivo ):
    
    urna = []
    with open( archivo ) as fid:
        spamreader = csv.reader(fid)
        ln = 1
        for row in spamreader:
            if ln <= 9:
                print( ln, ' : ', row )
            urna.append( float(row[0]) )
            ln = ln+1
    
    return np.array( urna, dtype=np.float64 )


def distanciaPoroMasCercano( poro, coordenadas ):

    distMin = math.sqrt(math.fsum((coordenadas[poro]-coordenadas[0])**2)) 

    NPT = len( coordenadas )

    for i in range( 1, NPT ):

        distancia = math.sqrt(math.fsum((coordenadas[poro]-coordenadas[i])**2))

        if distancia < distMin:
            
            distMin = distancia
            
    return distMin

def buscarConexion( arista, conexiones ):
    
    inicio = arista[0]
    fin = arista[1]
    NTC = len( conexiones )
    for i in range(NTC):
        if ( ( inicio == conexiones[i][0] ) and ( fin == conexiones[i][1] ) ) or ( ( inicio == conexiones[i][1] ) and ( fin == conexiones[i][0] ) ):
            # print( conexiones[i] )
            break
            
    return i

def buscarRuta( rutaNodos, conexiones ):
    
    ruta = np.zeros( (len(conexiones)), dtype=bool ) # False
    NC = len( rutaNodos )-1
    for i in range(NC):
        arista = [ rutaNodos[i], rutaNodos[i+1] ]
        j = buscarConexion( arista, conexiones )
        ruta[j] = True
        
    return ruta
    
                  
         
    
    
                
            
            


    
    
