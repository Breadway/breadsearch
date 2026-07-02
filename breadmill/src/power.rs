use std::fs;
use std::path::Path;

/// Best-effort check for whether the system is currently on AC/mains power.
/// Scans `/sys/class/power_supply` for a Mains or USB supply with `online=1`.
/// Systems with no such supply at all (desktops, no battery) are treated as
/// always on power, so this never blocks indexing on hardware without a
/// battery to protect.
pub fn on_ac_power() -> bool {
    let dir = Path::new("/sys/class/power_supply");
    let Ok(entries) = fs::read_dir(dir) else {
        return true;
    };

    let mut found_mains = false;
    for entry in entries.flatten() {
        let path = entry.path();
        let supply_type = fs::read_to_string(path.join("type")).unwrap_or_default();
        let supply_type = supply_type.trim();
        if supply_type != "Mains" && supply_type != "USB" {
            continue;
        }
        found_mains = true;
        let online = fs::read_to_string(path.join("online")).unwrap_or_default();
        if online.trim() == "1" {
            return true;
        }
    }

    !found_mains
}
