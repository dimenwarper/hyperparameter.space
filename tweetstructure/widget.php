<!DOCTYPE html>
<html>
<head>
<script src="https://unpkg.com/ngl"></script>
<style>
html, body { width: 100%; height: 100%; overflow: hidden; margin: 0; padding: 0; }
</style>
</head>
<body>
<div id="viewport" style="width:100%; height:100%;"></div>
<script>
var pdbid = "<?php echo $_GET['pdbid']; ?>"
//var pdbid = "3A6P"

// Setup to load data from rawgit
NGL.DatasourceRegistry.add(
    "data", new NGL.StaticDatasource( "//cdn.rawgit.com/arose/ngl/v2.0.0-dev.32/data/" )
);

// Create NGL Stage object
var stage = new NGL.Stage( "viewport" );

// Handle window resizing
window.addEventListener( "resize", function( event ){
    stage.handleResize();
}, false );


// Code for example: component/matrix

// Load a protein
stage.loadFile("https://files.rcsb.org/download/" + pdbid + ".pdb").then(function (o) {
  o.addRepresentation("cartoon", { color: "resname" })
  o.addRepresentation("base", { color: "resname" })
  o.addRepresentation("ball+stick", { color: "resname", visible: false })
  stage.autoView()
})


</script>
</body>
</html>
