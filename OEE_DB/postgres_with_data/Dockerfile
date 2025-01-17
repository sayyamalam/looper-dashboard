# postgres_with_data/Dockerfile
FROM postgres:15

# Kopiere dein init.sql + CSV in /docker-entrypoint-initdb.d/
COPY init.sql /docker-entrypoint-initdb.d/
COPY 2024_11_12_auszug.csv /docker-entrypoint-initdb.d/
