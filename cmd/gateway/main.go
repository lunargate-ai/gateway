package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/lunargate-ai/gateway/internal/api"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelselect"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/remotecontrol"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/internal/updatecheck"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var version = "dev"

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	logLevel := flag.String("log-level", "", "Override log level (debug, info, warn, error)")
	flag.Parse()

	// --- Load Config ---
	cfgManager, err := config.NewManager(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	cfg := cfgManager.Get()
	effectiveLogging, err := resolveLoggingConfig(cfg.Logging, *logLevel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid logging configuration: %v\n", err)
		os.Exit(1)
	}

	// --- Setup Logging ---
	effectiveLogLevel := setupLogging(effectiveLogging)
	startupLogFormat := effectiveLogging.Format
	printStartupBanner(version, *configPath, effectiveLogLevel, cfg.Server.Address())

	log.Info().
		Str("version", version).
		Str("config", *configPath).
		Str("log_level", effectiveLogLevel).
		Msg("starting LunarGate gateway")

	// --- Initialize Components ---
	registry := providers.NewRegistry(cfg.Providers)
	if len(registry.List()) == 0 {
		log.Fatal().Msg("no providers configured - at least one provider is required")
	}

	routingEngine := routing.NewEngine(cfg.Routing)
	retrier := resilience.NewRetrier(cfg.Retry)
	cbManager := resilience.NewCircuitBreakerManager()
	fallbackExec := resilience.NewFallbackExecutor(retrier, cbManager)
	cache := middleware.NewCache(cfg.Cache)
	rateLimiter := middleware.NewRateLimiter(cfg.RateLimit)
	authManager, err := security.NewManager(cfg.Security)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to initialize inbound auth")
	}
	streamer := streaming.NewHandler()
	metrics := observability.NewMetrics()
	healthChecker := health.NewChecker(version)
	collectorClient := observability.NewCollectorClient(cfg.General, cfg.DataSharing, version)
	updateChecker := updatecheck.NewChecker(cfg.UpdateCheck, version)
	updateCheckCtx, updateCheckCancel := context.WithCancel(context.Background())
	updateChecker.Start(updateCheckCtx)
	log.Info().
		Bool("enabled", cfg.UpdateCheck.Enabled).
		Str("endpoint", redactedHTTPURL(cfg.UpdateCheck.Endpoint)).
		Msg("automatic update check status")
	selector := modelselect.NewEngine(cfg.ModelSelect)
	store := modelstore.NewStore(registry, cfg.Providers)
	handler := api.NewHandler(registry, routingEngine, fallbackExec, cache, streamer, metrics, collectorClient, selector, store)
	remoteControlAddress, remoteControlAddressErr := localLoopbackAddress(cfg.Server)
	remoteControlBaseURL := ""
	if remoteControlAddressErr == nil {
		remoteControlBaseURL = "http://" + remoteControlAddress
	}
	var remoteControlMu sync.Mutex
	var remoteControlClient *remotecontrol.Client
	remoteControlCancel := func() {}
	refreshRemoteControlHello := func() {
		remoteControlMu.Lock()
		defer remoteControlMu.Unlock()
		if remoteControlClient != nil {
			remoteControlClient.RefreshHello()
		}
	}
	reconcileRemoteControl := func(cfg *config.Config) {
		remoteControlMu.Lock()
		defer remoteControlMu.Unlock()

		remoteControlCancel()
		remoteControlCancel = func() {}
		remoteControlClient = nil
		if cfg.DataSharing.Enabled && cfg.DataSharing.RemoteControl && remoteControlAddressErr != nil {
			log.Error().Err(remoteControlAddressErr).Msg("remote control disabled: secure local sandbox address unavailable")
		} else {
			remoteControlClient = remotecontrol.NewClient(
				cfg.General,
				cfg.DataSharing,
				cfg.Security,
				version,
				remoteControlBaseURL,
				handler.RuntimeRouteNames,
				handler.RuntimeModelSnapshotIDs,
				handler.RuntimeModelIDs,
			)
		}
		if remoteControlClient != nil {
			rcCtx, cancel := context.WithCancel(context.Background())
			remoteControlCancel = cancel
			remoteControlClient.Start(rcCtx)
		}

		logRemoteControlStatus(cfg, remoteControlClient)
	}
	currentCfg := cfg
	reconcileRemoteControl(cfg)

	// --- Setup Hot-Reload ---
	cfgManager.OnChange(func(newCfg *config.Config) {
		oldCfg := currentCfg
		currentCfg = newCfg
		generalChanged := !reflect.DeepEqual(oldCfg.General, newCfg.General)
		dataSharingChanged := !reflect.DeepEqual(oldCfg.DataSharing, newCfg.DataSharing)
		securityChanged := !reflect.DeepEqual(oldCfg.Security, newCfg.Security)
		serverChanged := !reflect.DeepEqual(oldCfg.Server, newCfg.Server)
		updateCheckChanged := !reflect.DeepEqual(oldCfg.UpdateCheck, newCfg.UpdateCheck)
		loggingChanged := !reflect.DeepEqual(oldCfg.Logging, newCfg.Logging)

		if serverChanged {
			log.Warn().Msg("server config changed; listen address and timeouts still require process restart to fully apply")
		}

		if loggingChanged {
			reloadedLogging, loggingErr := resolveLoggingConfig(newCfg.Logging, *logLevel)
			if loggingErr != nil {
				log.Error().Err(loggingErr).Msg("failed to reconcile logging config; keeping previous log level")
			} else {
				setLogLevel(reloadedLogging.Level)
				if reloadedLogging.Format != startupLogFormat {
					log.Warn().
						Str("configured_format", reloadedLogging.Format).
						Str("active_format", startupLogFormat).
						Msg("logging format changed; process restart required to apply")
				}
			}
		}
		runtimeChanged, runtimeErr := handler.UpdateRuntime(newCfg.Providers, newCfg.Routing, newCfg.ModelSelect)
		if runtimeErr != nil {
			log.Error().Err(runtimeErr).Msg("failed to reconcile runtime generation; keeping previous runtime")
		}
		rateLimiter.UpdateConfig(newCfg.RateLimit)
		cache.UpdateConfig(newCfg.Cache)
		retrier.UpdateConfig(newCfg.Retry)
		collectorClient.UpdateConfig(newCfg.General, newCfg.DataSharing)
		if updateCheckChanged {
			updateChecker.UpdateConfig(newCfg.UpdateCheck)
			log.Info().
				Bool("enabled", newCfg.UpdateCheck.Enabled).
				Str("endpoint", redactedHTTPURL(newCfg.UpdateCheck.Endpoint)).
				Msg("automatic update check config updated")
		}
		if securityChanged {
			if err := authManager.UpdateConfig(newCfg.Security); err != nil {
				log.Error().Err(err).Msg("failed to reconcile inbound auth config; keeping previous auth state")
			}
		}

		if dataSharingChanged || generalChanged || securityChanged {
			reconcileRemoteControl(newCfg)
		} else if runtimeChanged {
			refreshRemoteControlHello()
			remoteControlMu.Lock()
			clientSnapshot := remoteControlClient
			remoteControlMu.Unlock()
			logRemoteControlStatus(newCfg, clientSnapshot)
		}
		log.Info().Msg("hot-reload: routing, providers, retry, cache, rate limit, inbound auth, model selection, collector, and remote control reconciled")
	})
	cfgManager.WatchChanges()

	// --- Create API Handler & Router ---
	router := api.NewRouter(handler, authManager, rateLimiter, healthChecker)

	// --- Start HTTP Server ---
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.Server.ReadTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
		IdleTimeout:       cfg.Server.IdleTimeout,
		MaxHeaderBytes:    1 << 20,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Info().
			Str("address", cfg.Server.Address()).
			Strs("providers", handler.RuntimeProviderNames()).
			Msg("gateway listening")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("server failed")
		}
	}()

	<-done
	log.Info().Msg("shutting down gateway...")

	healthChecker.SetReady(false)
	remoteControlMu.Lock()
	remoteControlCancel()
	remoteControlMu.Unlock()
	updateCheckCancel()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := handler.CloseResponsesWebSockets(ctx); err != nil {
		log.Error().Err(err).Msg("responses websocket shutdown error")
	}
	if err := srv.Shutdown(ctx); err != nil {
		log.Error().Err(err).Msg("server shutdown error")
	}
	if collectorClient != nil {
		collectorClient.Stop()
	}
	if cache != nil {
		cache.Stop()
	}

	log.Info().Msg("gateway stopped")
}

func setupLogging(cfg config.LoggingConfig) string {
	return setupLoggingOutput(cfg, os.Stderr)
}

func setupLoggingOutput(cfg config.LoggingConfig, output io.Writer) string {
	zerolog.TimeFieldFormat = time.RFC3339Nano
	if cfg.Format == "console" {
		output = zerolog.ConsoleWriter{Out: output, TimeFormat: time.RFC3339}
	}
	log.Logger = zerolog.New(output).With().Timestamp().Logger()
	return setLogLevel(cfg.Level)
}

func setLogLevel(raw string) string {
	level, err := zerolog.ParseLevel(raw)
	if err != nil {
		level = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(level)

	return level.String()
}

func resolveLoggingConfig(cfg config.LoggingConfig, override string) (config.LoggingConfig, error) {
	cfg.Level = strings.ToLower(strings.TrimSpace(cfg.Level))
	cfg.Format = strings.ToLower(strings.TrimSpace(cfg.Format))
	if override = strings.ToLower(strings.TrimSpace(override)); override != "" {
		if _, err := zerolog.ParseLevel(override); err != nil {
			return config.LoggingConfig{}, fmt.Errorf("invalid --log-level value")
		}
		cfg.Level = override
	}
	return cfg, nil
}

func printStartupBanner(version, configPath, logLevel, address string) {
	fmt.Fprintf(os.Stderr, `
 _                           ____       _
| |   _   _ _ __   __ _ _ _ / ___| __ _| |_ ___
| |  | | | | '_ \ / _`+"`"+` | '_| |  _ / _`+"`"+` | __/ _ \
| |__| |_| | | | | (_| | | | |_| | (_| | ||  __/
|_____\__,_|_| |_|\__,_|_|  \____|\__,_|\__\___|

version: %s
config:  %s
log:     %s
listen:  %s

`, version, configPath, logLevel, address)
}

func localLoopbackAddress(cfg config.ServerConfig) (string, error) {
	host := strings.TrimSpace(cfg.Host)
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	}

	switch {
	case host == "", host == "0.0.0.0":
		host = "127.0.0.1"
	case host == "::":
		host = "::1"
	case strings.EqualFold(host, "localhost"):
		host = "127.0.0.1"
	default:
		ip := net.ParseIP(host)
		if ip == nil {
			return "", fmt.Errorf("server.host is not a wildcard or loopback address")
		}
		switch {
		case ip.IsUnspecified() && ip.To4() != nil:
			host = "127.0.0.1"
		case ip.IsUnspecified():
			host = "::1"
		case ip.IsLoopback():
			host = ip.String()
		default:
			return "", fmt.Errorf("server.host is not a wildcard or loopback address")
		}
	}
	return net.JoinHostPort(host, strconv.Itoa(cfg.Port)), nil
}

func logRemoteControlStatus(cfg *config.Config, remoteControlClient *remotecontrol.Client) {
	remoteControlEnabled := cfg.DataSharing.RemoteControl && remoteControlClient != nil
	remoteControlInstanceID := ""
	if remoteControlClient != nil {
		remoteControlInstanceID = remoteControlClient.InstanceID()
	}

	log.Info().
		Bool("data_sharing_enabled", cfg.DataSharing.Enabled).
		Bool("remote_control_enabled", remoteControlEnabled).
		Str("instance_id", remoteControlInstanceID).
		Str("backend_url", redactedHTTPURL(cfg.General.BackendURL)).
		Msg("gateway data sharing and remote control status")
}

func redactedHTTPURL(raw string) string {
	if redacted, ok := safeurl.RedactedHTTPURL(raw); ok {
		return redacted
	}
	return "[invalid]"
}
