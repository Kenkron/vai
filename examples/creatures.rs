#![allow(clippy::needless_return)]

extern crate rand;

use rand::random;
use macroquad::prelude::*;
use vai::VAID;

const FIRST_NAMES: [&'static str; 20] = [
    "Olivia", "Noah", "Emma", "Liam", "Amelia",
    "Oliver", "Sophia", "Elijah", "Charlotte", "Mateo",
    "Ava", "Lucas", "Isabella", "Levi", "Mia",
    "Leo", "Luna", "Ezra", "Evelyn", "Luca"];

const LAST_NAMES: [&'static str; 10] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"];

#[repr(usize)]
enum Senses {
    Bias,
    Hunger,
    FoodNorth,
    FoodSouth,
    FoodEast,
    FoodWest,
    Size
}

enum Actions {
    X,
    Y,
    Size
}

const MEMORIES: usize = 1;
const OUTPUT_NEURONS: usize = Actions::Size as usize + MEMORIES;
const INPUT_NEURONS: usize = Senses::Size as usize + OUTPUT_NEURONS;
const HIDDEN_NEURONS: usize = (INPUT_NEURONS + OUTPUT_NEURONS) / 2;

const TURTLE_SPEED: f32 = 0.05;
const SQUIRREL_SPEED: f32 = 0.1;

#[derive(Clone)]
struct Creature {
    uid: u64,
    first_name: &'static str,
    last_name: &'static str,
    position: Vec2,
    ai: VAID,
    hunger: f32,
    inputs: [f32; INPUT_NEURONS],
    outputs: [f32; OUTPUT_NEURONS]
}

struct World {
    pub size: Vec2,
    pub squirrel_texture: Texture2D,
    pub turtle_texture: Texture2D,
    pub nut_texture: Texture2D,
    pub squirrels: Vec<Box<Creature>>,
    pub turtles: Vec<Box<Creature>>,
    pub nuts: Vec<Vec2>
}

impl World {
    async fn new() -> World {
        Self {
            size: vec2(800., 600.),
            squirrel_texture: load_texture("examples/assets/chipmunk.png").await.unwrap(),
            turtle_texture: load_texture("examples/assets/turtle.png").await.unwrap(),
            nut_texture: load_texture("examples/assets/peanut.png").await.unwrap(),
            squirrels: vec![],
            turtles: vec![],
            nuts: vec![]
        }
    }
    fn distance_squared(&self, a: &Vec2, b: &Vec2) {
        let dx = (a.x - b.x).rem_euclid(self.size.x);
        let dx = dx.min(self.size.x - dx);
        let dy = (a.y - b.y).rem_euclid(self.size.y);
        let dy = dy.min(self.size.y - dy);
        dx * dx + dy * dy
    }
    fn world_to_viewport(&self, viewport: &Rect, point: &Vec2) -> Vec2 {
        let scale = vec2(screen_width() / viewport.w, screen_height() / viewport.h);
        vec2((point.x - viewport.x).rem_euclid(self.size.x) * scale.x,
            (point.y - viewport.y).rem_euclid(self.size.y) * scale.y)
    }
    fn update(&mut self) {
        let scent = |a: &Vec2, b: &Vec2|
            1./f32::max(self.distance_squared(a, b), 1.);
        let adjacent = |start: &Vec2| (
            *start + vec2(0., -1.), *start + vec2(0., 1.),
            *start +vec2(1., 0.), *start + vec2(-1., 0.));

        // Update AI
        self.turtles.iter_mut().for_each(|turtle| {
            use Senses::*;
            turtle.inputs[Bias as usize] = 1.;
            turtle.inputs[Hunger as usize] = turtle.hunger;
            let (north, south, east, west) = adjacent(&turtle.position);
            for nut in &self.nuts {
                turtle.inputs[FoodNorth as usize] += scent(&nut, &north) * 100.;
                turtle.inputs[FoodSouth as usize] += scent(&nut, &south) * 100.;
                turtle.inputs[FoodEast as usize] += scent(&nut, &east) * 100.;
                turtle.inputs[FoodWest as usize] += scent(&nut, &west) * 100.;
            }
            turtle.inputs[Size as usize..INPUT_NEURONS].clone_from_slice(&turtle.outputs);
            turtle.outputs.clone_from_slice(&turtle.ai.process_slice(&turtle.inputs));
        });

        // Update Actions
        let update_creature = |turtle: &mut Box<Creature>| {
            use Actions::*;
            let movement = vec2(turtle.outputs[X as usize], turtle.outputs[Y as usize]);
            turtle.position += movement.normalize_or_zero() * TURTLE_SPEED;
            turtle.position.x = turtle.position.x.rem_euclid(self.size.x);
            turtle.position.y = turtle.position.y.rem_euclid(self.size.y);
        };
        self.turtles.iter_mut().for_each(update_creature);
        self.squirrels.iter_mut().for_each(update_creature);
    }
    fn render(&self, viewport: &Rect) {
        let scale = vec2(screen_width() / viewport.w, screen_height() / viewport.h);
        let icon_size = vec2(1.,1.);
        let scaled_params = DrawTextureParams {
            dest_size: Some(scale),
            ..Default::default()
        };

        let draw_icon = |&icon, location: &Vec2|
            draw_texture_ex(icon,
                location.x - scale.x * icon_size.x * 0.5,
                location.y - scale.y * icon_size.y * 0.5,
                WHITE,
                scaled_params.clone());

        let draw_name = |&first, &last, location: &Vec2| {
            draw_text(first, location.x + icon_size.x * 0.5 * scale.x, location.y, 0.5 * scale.y, WHITE);
            draw_text(last,location.x + icon_size.x * 0.5 * scale.x, location.y + 0.5 * scale.y, 0.5 * scale.y, WHITE);};

        for nut in &self.nuts {
            draw_icon(&self.nut_texture, &self.world_to_viewport(viewport, nut));
        }
        for turtle in &self.turtles {
            let position = screen_position(&turtle.position);
            draw_icon(&self.turtle_texture, &position);
            draw_name(&turtle.first_name, &turtle.last_name, &position);
        }
        for squirrel in &self.squirrels {
            let position = screen_position(&squirrel.position);
            draw_icon(&self.squirrel_texture, &position);
            draw_name(&squirrel.first_name, &squirrel.last_name, &position);
        }
        //draw_icon(&self.nut_texture, &vec2(viewport.x + 1., viewport.y + 1.));
        //draw_icon(&self.nut_texture, &vec2(viewport.x + viewport.w, viewport.y+viewport.h));

    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "OMG! Evolution!!!".to_owned(),
        fullscreen: false,
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut id_counter: u64 = 0;
    let mut get_uid = || {id_counter += 1; id_counter};
    let mut world = World::new().await;
    let random_vec = || world.size * vec2(random(), random());
    let mut new_creature = || Creature {
        uid: get_uid(),
        first_name: FIRST_NAMES[random::<usize>() % FIRST_NAMES.len()],
        last_name: LAST_NAMES[random::<usize>() % LAST_NAMES.len()],
        position: random_vec(),
        ai: VAID::new(&[INPUT_NEURONS, HIDDEN_NEURONS,OUTPUT_NEURONS]).create_variant(1.),
        hunger: 0.,
        inputs: [1.; INPUT_NEURONS],
        outputs: [0.; OUTPUT_NEURONS]};
    for _ in 0..100 {
        world.nuts.push(random_vec());
        world.squirrels.push(Box::new(new_creature()));
        world.turtles.push(Box::new(new_creature()));
    }
    let mut viewport_center = world.size * 0.5;
    let mut viewport_scale = 0.1;
    let mut viewport = Rect{x: viewport_center.x, y: viewport_center.y};
    let mouse_vec = || vec2(mouse_position().0, mouse_position().1);
    let mut previous_mouse_position = mouse_vec();
    let mut selected_creature = Option::<Creature>::None;
    loop {
        // mouse_delta_position is only accurate once, and must be called each frame
        let mouse_delta = mouse_vec() - previous_mouse_position;
        previous_mouse_position = mouse_vec();

        // Update the world
        world.update();
        world.render(&viewport);

        // Allow user to select a creature for further information
        let mut selection_update = Option::<Creature>::None;
        if is_key_pressed(KeyCode::E) {
            selected_creature = None;
        }
        if let Some(selection) = &selected_creature {
            let mut check_for_selection = |creature_list: &Vec<Box<Creature>>| {
                for creature in creature_list {
                    if creature.uid == selection.uid {
                        selection_update = Some(*creature.clone());
                        break;
                    }
                }
            };
            check_for_selection(&world.turtles);
            check_for_selection(&world.squirrels);
        }
        if let Some(update) = selection_update {
            selected_creature = Some(update);
        }

        //UI and input
        let mut input_consumed = false;
        if let Some(selection) = &selected_creature {
            let ui_region = Rect{x: 0., y: 0., w: 300, h: screen_height()};
            let neurons = selection.ai.process_slice_transparent(&selection.inputs);
            draw_rect(ui_region.x, ui_region.y, ui_region.w, ui_region.h, BLACK);
            // Render circles based on float values
            for (layer_index, &layer) in neurons.iter().enumerate() {
                for (index, &value) in layer_index.iter.enumerate() {
                    let circle_color = value_to_color(value);

                    // Draw the circle
                    draw_circle(
                        ((layer_index + 1) * 20) as f32,
                        ((index + 1) * 20) as f32,
                        10,
                        color::new(-value.max(0.0), value.max(0.0), 1.0),
                    );
                }
            }
            if ui_region.contains(mouse_vec) {
                input_consumed = true;
            }
        }
        if !input_consumed && is_mouse_button_pressed(MouseButton::Left) {
            //for
        }
        if !input_consumed {
            // Update click + drag for view
            if is_mouse_button_down(MouseButton::Left) {
                viewport_center -= mouse_delta * viewport_scale;
            }
            let (_mouse_wheel_x, mouse_wheel_y) = mouse_wheel();
            viewport_scale *= 0.9_f32.powf(mouse_wheel_y);
            let viewport = Rect {
                x: viewport_center.x - 0.5 * screen_width() * viewport_scale,
                y: viewport_center.y - 0.5 * screen_height() * viewport_scale,
                w: screen_width() * viewport_scale,
                h: screen_height() * viewport_scale
            };
        }
        next_frame().await
    }
}
