
// plotar a rede com cada nó inicializado com a distancia entre os nós da próxima camada, por exemplo:

// x1c0 ------1------x1c0
// x1c1-------2------x1c0
// x1c2-------3------x1c0

// conforme os pesos vão se ajustando, pesos maiores são indicativos de menos distância entre os dois nós, ou seja a distancia entre os nós é 
// inversamente proporcional ao peso que cada nó tem para cada nó da camada seguinte.




//2d plot 

use plotters::pRelude::*;

fn project_3d_to_2d(x: f64, y: f64, z: f64) -> (f64, f64) {
    // Simple orthographic projection
    (x + z, y + z)
}

fn plot_network(nodes: &[(f64, f64, f64)], edges: &[((usize, usize), f64)], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Neural Network Visualization", ("sans-serif", 30).into_font())
        .build_cartesian_2d(-10.0..10.0, -10.0..10.0)?;

    // Draw edges
    for &((i, j), weight) in edges {
        let (x1, y1) = project_3d_to_2d(nodes[i].0, nodes[i].1, nodes[i].2);
        let (x2, y2) = project_3d_to_2d(nodes[j].0, nodes[j].1, nodes[j].2);
        chart.draw_series(LineSeries::new(
            vec![(x1, y1), (x2, y2)],
            &BLUE.mix(weight as f32),
        ))?;
    }

    // Draw nodes
    for (i, &(x, y, z)) in nodes.iter().enumerate() {
        let (x_proj, y_proj) = project_3d_to_2d(x, y, z);
        chart.draw_series(PointSeries::of_element(
            vec![(x_proj, y_proj)],
            5,
            &RED,
            &|c, s, st| {
                EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
            },
        ))?;
    }

    Ok(())
}

fn main() {
    let nodes = vec![
        (0.0, 0.0, 0.0), // Layer 1
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0), // Layer 2
        (2.0, 1.0, 1.0),
    ];
    let edges = vec![
        ((0, 3), 0.5),
        ((1, 3), 0.7),
        ((2, 3), 0.3),
        ((0, 4), 0.2),
        ((1, 4), 0.4),
    ];
    plot_network(&nodes, &edges, "network.png").unwrap();
}



// 3d plot

[dependencies]
kiss3d = "0.30"

//

use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::nalgebra::{Point3, Vector3};

fn main() {
    let mut window = Window::new("Neural Network Visualization");

    // Nodes
    let nodes = vec![
        Point3::new(0.0, 0.0, 0.0), // Layer 1
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 1.0), // Layer 2
        Point3::new(2.0, 1.0, 1.0),
    ];

    // Edges
    let edges = vec![
        ((0, 3), 0.5),
        ((1, 3), 0.7),
        ((2, 3), 0.3),
        ((0, 4), 0.2),
        ((1, 4), 0.4),
    ];

    // Draw nodes
    for node in &nodes {
        window.add_sphere(0.1).set_color(1.0, 0.0, 0.0).set_local_translation(*node);
    }

    // Draw edges
    for &((i, j), weight) in &edges {
        let start = nodes[i];
        let end = nodes[j];
        let line = window.add_line(start, end);
        line.set_color(0.0, 0.0, 1.0).set_line_width(weight as f32 * 5.0);
    }

    window.set_light(Light::StickToCamera);

    while window.render() {
        // Interactive loop
    }
}



use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::nalgebra::{Point3, Vector3};

fn main() {
    let mut window = Window::new("Neural Network Visualization");

    // Nodes
    let nodes = vec![
        Point3::new(0.0, 0.0, 0.0), // Layer 1
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 1.0), // Layer 2
        Point3::new(2.0, 1.0, 1.0),
    ];

    // Edges
    let edges = vec![
        ((0, 3), 0.5),
        ((1, 3), 0.7),
        ((2, 3), 0.3),
        ((0, 4), 0.2),
        ((1, 4), 0.4),
    ];

    // Draw nodes
    for node in &nodes {
        window.add_sphere(0.1).set_color(1.0, 0.0, 0.0).set_local_translation(*node);
    }

    // Draw edges
    for &((i, j), weight) in &edges {
        let start = nodes[i];
        let end = nodes[j];
        let line = window.add_line(start, end);
        line.set_color(0.0, 0.0, 1.0).set_line_width(weight as f32 * 5.0);
    }

    window.set_light(Light::StickToCamera);

    while window.render() {
        // Interactive loop
    }
}