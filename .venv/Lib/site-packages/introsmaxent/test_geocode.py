import geocode as gc

def main():
    gecoder = gc.GeoCode()
    gecoder.setfile('counties_curated.csv')

    toconvert = 'magnoliaweakley2014.csv'
    todisk = 'magtest.csv'

    gecoder.consume()

    fulljoin = gecoder.georefjoin(toconvert, ['stateProvince', 'county'],
                                  ['decimalLatitude','decimalLongitude'], todisk, True)
    print fulljoin

if __name__ == "__main__":
    main()